"""
MLX language model wrapper.

This module provides a wrapper class for MLX language models with text generation,
streaming, and caching capabilities.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import redirect_stderr, redirect_stdout
import gc
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generate, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import load
from outlines.processors import OutlinesLogitsProcessor

from ..const import DEFAULT_CONTEXT_LENGTH, DEFAULT_TRUST_REMOTE_CODE
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
DEFAULT_MIN_P = float(os.getenv("DEFAULT_MIN_P", "0.0"))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", "0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8192"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))


class MLX_LM:
    """
    A wrapper class for MLX Language Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
    ) -> None:
        """
        Initialize the MLX_LM wrapper by loading the model and tokenizer from the given path.

        Loads the model and tokenizer, sets tokenizer-related attributes (pad_token_id, bos_token),
        records the model type and context length, and creates an OutlinesTransformerTokenizer
        wrapper for outline-aware generation. During model download/load, stdout and stderr are
        temporarily redirected to suppress low-level progress output that can raise BrokenPipeError.

        Parameters:
            model_path (str): Filesystem path or remote identifier for the model to load.
            context_length (int): Maximum key-value cache / context length to use for generation.
            trust_remote_code (bool): Whether to allow model/tokenizer code from remote sources.

        Raises:
            ValueError: If the model or tokenizer cannot be loaded.
        """
        try:
            # Some third-party download utilities (huggingface_hub + tqdm)
            # write progress to stderr which can raise BrokenPipeError when
            # the server's stderr is closed or wrapped by the runtime. To
            # avoid surfacing that low-level error during handler
            # initialization we temporarily redirect stderr to devnull while
            # the model is downloaded/loaded.
            with (
                Path(os.devnull).open("w") as _devnull,
                redirect_stdout(_devnull),
                redirect_stderr(_devnull),
            ):
                self.model, self.tokenizer, *_ = load(
                    model_path,
                    lazy=False,
                    tokenizer_config={"trust_remote_code": trust_remote_code},
                )
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.max_kv_size = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f"Error loading model: {e}") from e

    def _get_pad_token_id(self) -> int:
        """Get a safe pad token ID, falling back through options."""
        if self.pad_token_id is not None:
            return int(self.pad_token_id)
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        return 0

    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        return mx.mean(embeddings, axis=1)

    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        """
        Normalize each embedding vector to have L2 norm equal to 1.

        Parameters:
            embeddings (mx.array): 2-D array of shape (N, D) containing N embedding vectors of dimension D.

        Returns:
            mx.array: Array of the same shape as `embeddings` where each row has been divided by its L2 norm (a small epsilon is added to norms to avoid division by zero).
        """
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (l2_norms + 1e-8)

    def _batch_process(
        self,
        prompts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[list[int]]:
        """
        Tokenize prompts in batches and pad each sequence to the batch maximum length for model-ready input.

        Parameters:
            prompts: List of prompt strings to tokenize.
            batch_size: Number of prompts to process per batch; controls memory vs throughput.

        Returns:
            A list of token id sequences where each sequence is padded with a safe pad token id to match its batch's maximum length.
        """
        all_tokenized = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tokenized_batch = []

            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)

            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)

            # Get safe pad token ID
            pad_token_id = self._get_pad_token_id()

            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)

        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> mx.array:
        """Tokenize a single prompt efficiently."""
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)

    def get_model_type(self) -> str:
        """
        Get the model type identifier.

        Returns:
            The model type as a string.
        """
        return str(self.model_type)

    def get_embeddings(
        self,
        prompts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        *,
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Get embeddings for a list of prompts efficiently.

        Args:
            prompts: List of text prompts
            batch_size: Size of batches for processing
            normalize: Whether to apply L2-normalization to each pooled embedding.
                When True, normalized embeddings are returned. When False, raw pooled vectors are returned.

        Returns
        -------
            List of embeddings as lists of floats, with a one-to-one mapping to input prompts.
            Embeddings may be L2-normalized depending on the normalize parameter.
        """
        # Process in batches to optimize memory usage
        all_embeddings: list[list[float]] = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                tokenized_batch = self._batch_process(batch_prompts, batch_size)

                # Convert to MLX array for efficient computation
                tokenized_array = mx.array(tokenized_batch)

                try:
                    # Compute embeddings for batch
                    batch_embeddings = self.model.model(tokenized_array)
                    pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
                    if normalize:
                        pooled_embedding = self._apply_l2_normalization(pooled_embedding)
                    all_embeddings.extend(pooled_embedding.tolist())  # type: ignore[arg-type]
                finally:
                    # Explicitly free MLX arrays to prevent memory leaks
                    del tokenized_batch
                    del tokenized_array
                    if "batch_embeddings" in locals():
                        del batch_embeddings
                    if "pooled_embedding" in locals():
                        del pooled_embedding
                    # Force MLX garbage collection
                    mx.clear_cache()
                    gc.collect()
        except Exception:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

        return all_embeddings

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[str | Generator[Any, None, None], int]:
        """
        Generate a model response for a chat conversation.

        Parameters:
            messages (list[dict[str, str]]): Conversation messages. Each message should be a mapping with string fields such as "role" and "content".
            stream (bool): If True, return a generator that yields streamed generation chunks; if False, return the full generated string.
            **kwargs: Optional generation parameters:
                - temperature (float): Sampling temperature.
                - top_p (float): Nucleus sampling probability threshold.
                - top_k (int): Top-k sampling cutoff.
                - min_p (float): Minimum probability threshold for sampling.
                - seed (int): Random seed for generation.
                - max_tokens (int): Maximum number of tokens to generate.
                - chat_template_kwargs (dict): Additional keyword arguments passed to the chat template application.
                - repetition_penalty (float): Penalty applied to repeated tokens.
                - repetition_context_size (int): Context window size for repetition handling.
                - schema (Any): Optional JSON schema used to constrain generation via an outlines logits processor.

        Returns:
            tuple[str | Generator[Any, None, None], int]: A pair where the first element is either the generated text (when stream is False)
            or a generator yielding streamed chunks (when stream is True), and the second element is the prompt length in tokens.
        """
        # Set default parameters if not provided
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P),
        }

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )
        json_schema = kwargs.get("schema")
        if json_schema:
            logits_processors.append(
                OutlinesLogitsProcessor(  # type: ignore[abstract,call-arg]
                    schema=json_schema,
                    tokenizer=self.outlines_tokenizer,
                    tensor_library_name="mlx",
                ),
            )

        mx.random.seed(seed)
        prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        sampler = make_sampler(**sampler_kwargs)

        prompt_tokens = len(input_tokens)

        if not stream:
            return generate(
                self.model,
                self.tokenizer,
                input_tokens,
                sampler=sampler,
                max_tokens=max_tokens,
                prompt_cache=prompt_cache,
                logits_processors=logits_processors,
            ), prompt_tokens
        # Streaming mode: return generator of chunks
        return stream_generate(
            self.model,
            self.tokenizer,
            input_tokens,
            sampler=sampler,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            logits_processors=logits_processors,
        ), prompt_tokens
