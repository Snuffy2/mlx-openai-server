"""MLX vision-language model handler for multimodal chat completions."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator, Generator
import gc
from http import HTTPStatus
import time
from typing import Any, NoReturn
import uuid

from fastapi import HTTPException
from loguru import logger

from ..const import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_DISABLE_AUTO_RESIZE,
    DEFAULT_ENABLE_AUTO_TOOL_CHOICE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REASONING_PARSER,
    DEFAULT_TOOL_CALL_PARSER,
    DEFAULT_TRUST_REMOTE_CODE,
)
from ..core import AudioProcessor, ImageProcessor, VideoProcessor
from ..core.queue import RequestQueue
from ..models.mlx_vlm import MLX_VLM
from ..schemas.openai import (
    ChatCompletionContentPart,
    ChatCompletionContentPartImage,
    ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartText,
    ChatCompletionContentPartVideo,
    ChatCompletionRequest,
    EmbeddingRequest,
    UsageInfo,
)
from ..utils.errors import create_error_response
from .parser import ParserFactory


class MLXVLMHandler:
    """
    Handler class for making requests to the underlying MLX multimodal model service.

    Provides caching, concurrent image processing, audio processing, and robust error handling.
    """

    def __init__(
        self,
        model_path: str,
        *,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        max_workers: int = 4,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        disable_auto_resize: bool = DEFAULT_DISABLE_AUTO_RESIZE,
        enable_auto_tool_choice: bool = DEFAULT_ENABLE_AUTO_TOOL_CHOICE,
        tool_call_parser: str | None = DEFAULT_TOOL_CALL_PARSER,
        reasoning_parser: str | None = DEFAULT_REASONING_PARSER,
        trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
    ) -> None:
        """
        Create and configure an MLXVLMHandler for the given model path.
        
        Parameters:
            model_path (str): Filesystem path or identifier for the MLX vision-language model.
            context_length (int): Maximum token context length the model should use.
            max_workers (int): Maximum concurrent worker threads for media processors (image/audio/video).
            max_concurrency (int): Maximum number of simultaneous model inference requests allowed.
            disable_auto_resize (bool): If true, image inputs will not be automatically resized by the handler.
            enable_auto_tool_choice (bool): If true, allow the handler to automatically select tools when available.
            tool_call_parser (str | None): Identifier of the parser to extract tool call structures from model output.
            reasoning_parser (str | None): Identifier of the parser to extract structured reasoning from model output.
            trust_remote_code (bool): If true, permit loading model code from remote sources that may execute custom code.
        """
        self.model_path = model_path
        self.model = MLX_VLM(
            model_path,
            context_length=context_length,
            trust_remote_code=trust_remote_code,
        )
        self.image_processor = ImageProcessor(max_workers)
        self.audio_processor = AudioProcessor(max_workers)
        self.video_processor = VideoProcessor(max_workers)
        self.disable_auto_resize = disable_auto_resize
        self.model_created = int(time.time())  # Store creation time when model is loaded
        self.model_type = self.model.get_model_type()

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser

        # Initialize request queue for multimodal and text tasks
        # We use the same queue for both multimodal and text tasks for simplicity
        # and to ensure we don't overload the model with too many concurrent requests
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXVLMHandler with model path: {model_path}")
        if disable_auto_resize:
            logger.info("Auto-resize is disabled for image processing")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Return metadata for the available model(s) configured for this handler.
        
        Returns:
            list[dict[str, Any]]: A list containing a single metadata dictionary with keys:
                - `id`: model identifier (the configured model path)
                - `object`: the object type (set to `"model"`)
                - `created`: model creation timestamp or value from `self.model_created`
                - `owned_by`: owner label (set to `"local"`).
            Returns an empty list if an error occurs while retrieving the metadata.
        """
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                },
            ]
        except Exception as e:
            logger.error(f"Error getting models. {type(e).__name__}: {e}")
            return []

    def _create_parsers(self) -> tuple[Any | None, Any | None]:
        """
        Create appropriate parsers based on model type and available tools.

        Uses ParserFactory for centralized parser creation logic.

        Returns
        -------
            Tuple of (thinking_parser, tool_parser)
        """
        return ParserFactory.create_parsers(
            model_type=self.model_type,
            manual_reasoning_parser=self.reasoning_parser,
            manual_tool_parser=self.tool_call_parser,
        )

    async def initialize(self, queue_config: dict[str, Any] | None = None) -> None:
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {"max_concurrency": self.request_queue.max_concurrency}
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency", self.request_queue.max_concurrency),
            timeout=queue_config.get("timeout", self.request_queue.timeout),
            queue_size=queue_config.get("queue_size", self.request_queue.queue_size),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXVLMHandler and started request queue")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns
        -------
            int: The number of tokens.
        """
        if not text:
            return 0
        try:
            # Try to use tokenizer from processor if available
            if hasattr(self.model.processor, "tokenizer"):
                tokens = self.model.processor.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            # Fallback for some processors that might behave differently or don't expose tokenizer directly
            # This part depends on specific processor implementation, but usually they have a tokenizer
            if hasattr(self.model.processor, "encode"):
                tokens = self.model.processor.encode(text, add_special_tokens=False)
                return len(tokens)
            logger.warning("Could not find tokenizer in processor to count tokens")
        except Exception as e:
            logger.warning(f"Failed to count tokens. {type(e).__name__}: {e}")
        return 0

    def _count_message_tokens(self, messages: list[dict[str, Any]], **kwargs: Any) -> int:
        """
        Estimate the number of prompt tokens for a list of chat messages.
        
        This provides an approximate, text-only token count for multimodal prompts; token contributions from images, audio, or video are not reliably included and may be undercounted. The function uses the model processor's chat templating when available and falls back to a conservative text-only aggregation on error.
        
        Parameters:
            messages (list[dict[str, Any]]): Chat messages to count tokens for.
            **kwargs: Passed through to the processor's apply_chat_template call when available.
        
        Returns:
            int: Approximate number of prompt tokens (text-only for VLMs).
        """
        try:
            # We need to handle the fact that messages might contain images/audio which apply_chat_template might not handle directly
            # if we pass them as is, or it might handle them if they are formatted correctly.
            # MLX_VLM's apply_chat_template (via processor) usually expects text-only messages or handles them if configured.
            # However, looking at MLX_VLM.__call__, it calls self.processor.apply_chat_template with tokenize=False first.

            # Let's try to use the processor's apply_chat_template with tokenize=True if possible,
            # or tokenize=False and then encode.

            # For VLM, the prompt tokens also depend on images.
            # This is complex because image tokens depend on the model and how images are processed.
            # For now, we will try to approximate or use the processor if it supports it.

            # Simplification: We will use the same logic as in MLX_VLM.__call__ to get the text prompt,
            # and then encode it. This might miss image tokens if they are added separately.

            # NOTE: Accurate token counting for VLMs is tricky without running the full preparation pipeline.
            # We will try to get the text part at least.

            text = self.model.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )

            # Now encode the text
            return self._count_tokens(text)

        except Exception as e:
            logger.warning(f"Failed to count message tokens. {type(e).__name__}: {e}")
            # Fallback: rough estimate
            total_text = ""
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_text += content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            total_text += part.get("text", "")
            return self._count_tokens(total_text)

    async def generate_multimodal_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream multimodal chat completion chunks, parse reasoning and tool-call content when applicable, and yield parsed pieces followed by a final usage summary.
        
        Parameters:
            request (ChatCompletionRequest): Multimodal chat request; its `stream` flag will be set to True.
        
        Returns:
            dict[str, Any]: Yields dictionaries representing either parsed response pieces (e.g., reasoning or tool-call outputs, or {"content": "<text>"}) or a final {"__usage__": UsageInfo(...)} entry with token counts.
        
        Raises:
            HTTPException: Raised with status 429 when the service is at capacity; raised with status 500 on internal failures.
        """
        try:
            # Enforce streaming mode for queued multimodal requests to ensure
            # the underlying model returns a generator instead of a full string.
            request.stream = True

            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"

            request_dict = await self._prepare_multimodal_request(request)

            # Ensure the request data seen by _process_request is also marked as streaming
            request_dict["stream"] = True

            # Submit to the multimodal queue and get the generator
            response_generator, prompt_tokens = await self.request_queue.submit(
                request_id,
                request_dict,
            )

            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            chat_template_kwargs = request_dict.get("chat_template_kwargs", {})
            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                if not enable_thinking:
                    thinking_parser = None

            is_first_chunk = True
            completion_chunks = []  # Accumulate completion for token counting
            after_thinking_close_content = None

            # Process and yield each chunk asynchronously
            for chunk in response_generator:
                # Handle both string chunks and object chunks with .text attribute
                if isinstance(chunk, str):
                    text = chunk
                elif hasattr(chunk, "text") and chunk.text:
                    text = chunk.text
                else:
                    # Skip invalid/empty chunks
                    continue

                completion_chunks.append(text)
                if is_first_chunk:
                    if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(
                        self.reasoning_parser,
                    ):
                        text = thinking_parser.get_thinking_open() + text
                    is_first_chunk = False

                if thinking_parser:
                    parsed_content, is_complete = thinking_parser.parse_stream(text)
                    after_thinking_close_content = None
                    if parsed_content:
                        if isinstance(parsed_content, dict):
                            after_thinking_close_content = parsed_content.pop("content", None)
                        yield parsed_content
                    if is_complete:
                        thinking_parser = None
                    if after_thinking_close_content:
                        text = after_thinking_close_content
                    else:
                        continue

                if tool_parser:
                    parsed_content, _ = tool_parser.parse_stream(text)
                    if parsed_content:
                        yield parsed_content
                    continue

                yield {"content": text}

            # Count completion tokens and yield usage info at the end
            completion_text = "".join(completion_chunks)
            completion_tokens = self._count_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            }

        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=content) from None
        except HTTPException:
            # Preserve existing HTTP error semantics from request prep
            raise
        except Exception as e:
            logger.error(
                f"Error in multimodal stream generation for request {request_id}. {type(e).__name__}: {e}",
            )
            content = create_error_response(
                f"Failed to generate multimodal stream: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=content) from e

    async def generate_multimodal_response(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a completed multimodal chat response and token usage for the given request.
        
        Given a prepared multimodal chat request, produce the model's full (non-streaming) response and a UsageInfo object describing token consumption. If parsers are configured for reasoning or tool calls, the returned `response` will be a dictionary with `reasoning_content` (possibly redacted), `tool_calls` (parsed tool invocations), and `content` (final assistant text); otherwise `response` will be the raw assistant string.
        
        Parameters:
            request: ChatCompletionRequest containing the messages and any multimedia parts to include in the multimodal prompt.
        
        Returns:
            dict[str, Any]: A dictionary with two keys:
                - "response": either a raw response string or a dict with keys:
                    - "reasoning_content": parsed reasoning output or None,
                    - "tool_calls": parsed tool call structures or None,
                    - "content": final assistant message text.
                - "usage": a UsageInfo instance with `prompt_tokens`, `completion_tokens`, and `total_tokens`.
        """
        try:
            # Create a unique request ID
            request_id = f"multimodal-{uuid.uuid4()}"

            request_dict = await self._prepare_multimodal_request(request)

            response, prompt_tokens = await self.request_queue.submit(request_id, request_dict)
        except asyncio.QueueFull:
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail=content) from None
        except HTTPException:
            # Preserve existing HTTP error semantics from request prep
            raise
        except Exception as e:
            logger.error(f"Error in multimodal response generation. {type(e).__name__}: {e}")
            content = create_error_response(
                f"Failed to generate multimodal response: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=content) from e
        else:
            # Count completion tokens
            completion_tokens = self._count_tokens(response)
            total_tokens = prompt_tokens + completion_tokens

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            if not thinking_parser and not tool_parser:
                return {"response": response, "usage": usage}

            parsed_response = {"reasoning_content": None, "tool_calls": None, "content": None}
            response_text = response

            if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(
                self.reasoning_parser,
            ):
                response_text = thinking_parser.get_thinking_open() + response_text

            if thinking_parser:
                thinking_response, response_text = thinking_parser.parse(response_text)
                parsed_response["reasoning_content"] = thinking_response
            if tool_parser:
                tool_response, response_text = tool_parser.parse(response_text)
                parsed_response["tool_calls"] = tool_response
            parsed_response["content"] = response_text

            return {"response": parsed_response, "usage": usage}

    async def generate_embeddings_response(self, _request: EmbeddingRequest) -> NoReturn:
        """
        Generate embeddings for a given text input.

        This function always raises an HTTPException(400) as embeddings are not supported for VLM models.

        Args:
            _request: EmbeddingRequest object containing the text input.

        Raises
        ------
            HTTPException: Embeddings are not supported for VLM models
        """
        # Embeddings are not supported for VLM models
        content = create_error_response(
            "Embeddings are not supported for VLM models",
            "bad_request",
            HTTPStatus.BAD_REQUEST,
        )
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

    async def close(self) -> None:
        """Explicitly cleanup resources asynchronously."""
        if hasattr(self, "image_processor"):
            await self.image_processor.cleanup()
        if hasattr(self, "audio_processor"):
            await self.audio_processor.cleanup()
        if hasattr(self, "video_processor"):
            await self.video_processor.cleanup()

    async def cleanup(self) -> None:
        """
        Release handler resources and stop background processing before shutdown.
        
        Stops the request queue if present, invokes cleanup on image, audio, and video processors when available, and forces garbage collection to free remaining resources.
        """
        logger.info("Cleaning up MLXVLMHandler resources")
        if hasattr(self, "request_queue"):
            try:
                await self.request_queue.stop()
            except Exception as e:
                logger.error(f"Error stopping request queue: {e}")
        if hasattr(self, "image_processor"):
            try:
                await self.image_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up image processor: {e}")
        if hasattr(self, "audio_processor"):
            try:
                await self.audio_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio processor: {e}")
        if hasattr(self, "video_processor"):
            try:
                await self.video_processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up video processor: {e}")

        # Force garbage collection after cleanup
        gc.collect()
        logger.info("MLXVLMHandler cleanup completed successfully")

    async def _process_request(
        self,
        request_data: dict[str, Any],
    ) -> tuple[str | Generator[str, None, None], int]:
        """
        Process a multimodal request from the queue and return the model output and prompt token count.
        
        Parameters:
            request_data (dict[str, Any]): Request dictionary containing at least the keys "messages" (chat messages) and "stream" (bool), plus other model parameters forwarded to the model call.
        
        Returns:
            tuple[str | Generator[str, None, None], int]: 
                - First element: the model response — either a complete string (when stream is False) or a synchronous generator yielding response chunks (when stream is True).
                - Second element: the number of prompt tokens consumed by the request.
        """
        try:
            # Extract request parameters
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)

            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("messages", None)
            model_params.pop("stream", None)

            # Call the model
            response, _ = self.model(
                messages=messages,
                stream=stream,
                **model_params,
            )
            # Force garbage collection after model inference
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing multimodal request. {type(e).__name__}: {e}")
            # Clean up on error
            gc.collect()
            raise
        else:
            prompt_tokens = self._count_message_tokens(messages, **model_params)
            return response, prompt_tokens

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics from the request queue.

        Returns
        -------
            Dict with queue statistics.
        """
        return self.request_queue.get_queue_stats()

    async def _reformat_multimodal_content_part(
        self,
        content_part: ChatCompletionContentPart,
    ) -> dict[str, Any]:
        """
        Convert a ChatCompletionContentPart into a normalized dictionary describing its type and payload.
        
        Parameters:
            content_part (ChatCompletionContentPart): A multimodal message part (image, video, audio, or text) to normalize.
        
        Returns:
            dict[str, Any]: A dictionary with at least a `content_part` key whose value is a dict with a `type` field (`"image"`, `"video"`, or `"text"`).
                - For images: `content_part` contains `"image"` with the local file path and the top-level dict also includes `"path"` with the same path.
                - For videos: `content_part` contains `"video"` with the local file path and the top-level dict also includes `"path"`.
                - For text: `content_part` contains `"text"` with the textual content.
                - For unknown part types: `content_part` contains `"text"` with the string representation of the part.
        
        Raises:
            HTTPException: Raises a 400 Bad Request when an audio input part is provided (audio is not supported).
        """
        if (
            isinstance(content_part, ChatCompletionContentPartImage)
            and content_part.image_url is not None
        ):
            image_url = content_part.image_url.url
            # Validate base64 data URLs before processing
            self._validate_image_url(image_url)
            image_path = await self.image_processor.process_image_url(
                image_url,
                resize=not self.disable_auto_resize,
            )
            return {"content_part": {"type": "image", "image": image_path}, "path": image_path}

        if (
            isinstance(content_part, ChatCompletionContentPartInputAudio)
            and content_part.input_audio is not None
        ):
            content = create_error_response(
                "Audio input is not supported for VLM models",
                "bad_request",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        if (
            isinstance(content_part, ChatCompletionContentPartVideo)
            and content_part.video_url is not None
        ):
            video_url = content_part.video_url.url
            # Note: Video validation could be added here if needed
            video_path = await self.video_processor.process_video_url(video_url)
            return {
                "content_part": {
                    "type": "video",
                    "video": video_path,
                },
                "path": video_path,
            }

        if isinstance(content_part, ChatCompletionContentPartText):
            return {"content_part": {"type": "text", "text": content_part.text}}

        # Fallback for unknown types
        return {"content_part": {"type": "text", "text": str(content_part)}}

    async def _prepare_multimodal_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """
        Normalize chat messages and collect local media paths and model parameters for a multimodal inference request.
        
        Processes each incoming message to:
        - Preserve system, assistant, and tool messages as-is.
        - Convert user messages that are either plain text or lists of multimodal content parts into a normalized message list.
        - Reformat multimodal content parts into standardized content entries and collect local paths for images, audio, and video into separate lists.
        - Assemble chat_template_kwargs and model parameters (temperature, top_p, frequency_penalty, presence_penalty, max_tokens, stream) into the returned request dictionary.
        Raises an HTTPException with status 400 for invalid message content or when media processing fails.
        
        Parameters:
            request (ChatCompletionRequest): Incoming client request containing messages, tools, tool_choice, and model parameters.
        
        Returns:
            dict[str, Any]: A dictionary containing:
                - messages: List[dict] of normalized chat message objects ready for the model.
                - images: List[str] of local image file paths.
                - audios: List[str] of local audio file paths.
                - videos: List[str] of local video file paths.
                - temperature, top_p, frequency_penalty, presence_penalty, max_tokens: model generation parameters.
                - chat_template_kwargs: dict of template and tool-related settings.
                - stream: bool indicating whether the request should be streamed.
        """
        chat_messages = []
        images = []
        audios = []
        videos = []

        try:
            # Process each message in the request
            for message in request.messages:
                # Handle system and assistant messages (simple text content)
                if message.role in ["system", "assistant"]:
                    chat_messages.append({"role": message.role, "content": message.content})
                    continue

                # Handle user messages
                if message.role == "user":
                    # Case 1: Simple string content
                    if isinstance(message.content, str):
                        chat_messages.append({"role": "user", "content": message.content})
                        continue

                    # Case 2: Content is a list of dictionaries or objects
                    if isinstance(message.content, list):
                        formatted_content_parts = []

                        for content_part in message.content:
                            formatted_content_part = await self._reformat_multimodal_content_part(
                                content_part,
                            )
                            if isinstance(content_part, ChatCompletionContentPartImage):
                                images.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartInputAudio):
                                audios.append(formatted_content_part["path"])
                            elif isinstance(content_part, ChatCompletionContentPartVideo):
                                videos.append(formatted_content_part["path"])

                            formatted_content_parts.append(formatted_content_part["content_part"])
                        chat_messages.append({"role": "user", "content": formatted_content_parts})
                    else:
                        content = create_error_response(
                            "Invalid message content format",
                            "invalid_request_error",
                            HTTPStatus.BAD_REQUEST,
                        )
                        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)
                elif message.role == "tool":
                    chat_messages.append({"role": "tool", "content": message.content})
                    continue

            chat_template_kwargs = request.chat_template_kwargs.model_dump()
            request_dict = {
                "messages": chat_messages,
                "images": images,
                "audios": audios,
                "videos": videos,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "max_tokens": request.max_tokens,
                "chat_template_kwargs": chat_template_kwargs,
                "stream": request.stream,
            }

            tools = request.tools or None
            tool_choice = request.tool_choice or None

            if tools:
                # Enable auto tool choice if requested via CLI flag
                if self.enable_auto_tool_choice and tool_choice == "auto":
                    chat_template_kwargs["tool_choice"] = "auto"
                elif tool_choice:
                    logger.warning("Tool choice has not supported yet, will be ignored.")
                chat_template_kwargs["tools"] = tools

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare multimodal request. {type(e).__name__}: {e}")
            content = create_error_response(
                f"Failed to process request: {e}",
                "bad_request",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e
        else:
            return request_dict

    def _validate_image_url(self, url: str) -> None:
        """
        Validate that an image URL is non-empty and, if it is a data URL, that it encodes a valid base64 image.
        
        Parameters:
            url (str): The image URL or data URL to validate. Data URLs must start with "data:image/".
        
        Raises:
            HTTPException: If `url` is empty or a data URL is not a valid base64-encoded image (HTTP 400).
        """
        if not url:
            content = create_error_response(
                "Empty image URL provided",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        # Validate base64 images
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Invalid image format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(
                    f"Invalid base64 image: {e}",
                    "invalid_request_error",
                    HTTPStatus.BAD_REQUEST,
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e

    def _validate_audio_data(self, url: str) -> None:
        """
        Validate an audio data URL and ensure it is present and, if a data URL, correctly formatted base64 audio.
        
        Parameters:
            url (str): Audio data URL to validate; may be a normal URL or a data URL starting with `data:audio/...`.
        
        Raises:
            HTTPException: With status 400 if `url` is empty or if a `data:` URL is not a valid `data:audio/...` header or contains invalid base64 data.
        """
        if not url:
            content = create_error_response(
                "Empty audio data provided",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content)

        # Validate base64 audio
        if url.startswith("data:"):
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:audio/"):
                    raise ValueError("Invalid audio format")
                base64.b64decode(encoded)
            except Exception as e:
                content = create_error_response(
                    f"Invalid base64 audio: {e}",
                    "invalid_request_error",
                    HTTPStatus.BAD_REQUEST,
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e