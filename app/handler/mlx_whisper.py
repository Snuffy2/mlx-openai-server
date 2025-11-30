"""
Handler for MLX Whisper audio transcription models.

This module provides the MLXWhisperHandler class for processing audio transcription
requests using MLX Whisper models, with support for streaming responses and
request queuing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
import gc
from http import HTTPStatus
from pathlib import Path
import tempfile
import time
from typing import Any, cast
import uuid

from fastapi import HTTPException, UploadFile
from loguru import logger

from ..core.queue import RequestQueue
from ..models.mlx_whisper import MLX_Whisper
from ..schemas.openai import (
    Delta,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    TranscriptionResponseStream,
    TranscriptionResponseStreamChoice,
    TranscriptionUsageAudio,
)
from ..utils.errors import create_error_response


class MLXWhisperHandler:
    """
    Handler class for making requests to the underlying MLX Whisper model service.

    Provides request queuing, metrics tracking, and robust error handling for audio transcription.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1) -> None:
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the model directory.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path: str = model_path
        self.model: MLX_Whisper = MLX_Whisper(model_path)
        self.model_created: int = int(time.time())  # Store creation time when model is loaded

        # Initialize request queue for audio transcription tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXWhisperHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        List available models and their metadata.

        Returns:
            list[dict[str, Any]]: A list of model metadata dictionaries. Each dictionary contains:
                - `id`: model identifier (the configured model path)
                - `object`: object type (e.g., "model")
                - `created`: model creation timestamp
                - `owned_by`: owner identifier (e.g., "local")
        """
        return [
            {
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local",
            },
        ]

    async def initialize(self, queue_config: dict[str, Any] | None = None) -> None:
        """
        Initialize or reconfigure the request queue and start its worker loop.

        If `queue_config` is not provided, a default configuration is created with a longer audio-processing timeout (600 seconds), a default queue size of 50, and the handler's current `request_queue.max_concurrency`. The method replaces the handler's `request_queue` with a new RequestQueue configured from `queue_config` (falling back to existing queue attributes when present) and starts it using the handler's `_process_request` worker.

        Parameters:
            queue_config (dict[str, Any] | None): Optional queue configuration with keys
                `max_concurrency` (int), `timeout` (seconds), and `queue_size` (int).
                Missing keys are filled from existing queue attributes or sensible defaults.
        """
        if not queue_config:
            queue_config = {
                "max_concurrency": self.request_queue.max_concurrency,
                "timeout": 600,  # Longer timeout for audio processing
                "queue_size": 50,
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get(
                "max_concurrency",
                self.request_queue.max_concurrency if self.request_queue else 1,
            ),
            timeout=queue_config.get(
                "timeout",
                self.request_queue.timeout if self.request_queue else 600,
            ),
            queue_size=queue_config.get(
                "queue_size",
                self.request_queue.queue_size if self.request_queue else 50,
            ),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXWhisperHandler and started request queue")

    async def generate_transcription_response(
        self,
        request: TranscriptionRequest,
    ) -> TranscriptionResponse | str:
        """
        Generate a transcription for the provided request and return it in the requested format.

        Parameters:
            request (TranscriptionRequest): The transcription request containing the uploaded audio and output format preferences; if `request.response_format` equals `TranscriptionResponseFormat.JSON` the result is returned as a `TranscriptionResponse`, otherwise the function returns the transcribed text string.

        Returns:
            TranscriptionResponse | str: A `TranscriptionResponse` object when JSON format is requested, the transcribed text string otherwise.

        Notes:
            The temporary audio file created for processing is removed before the function returns.
        """
        request_id = f"transcription-{uuid.uuid4()}"
        temp_file_path = None

        try:
            request_data = await self.prepare_transcription_request(request)
            temp_file_path = request_data.get("audio_path")
            response = await self.request_queue.submit(request_id, request_data)
            duration_seconds = int(response.get("duration", 0))
            response_data = TranscriptionResponse(
                text=response["text"],
                usage=TranscriptionUsageAudio(type="duration", seconds=duration_seconds),
            )
            if request.response_format == TranscriptionResponseFormat.JSON:
                return response_data
            # Return plain text for text response format
            return response_data.text
        finally:
            # Clean up temporary file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
            # Force garbage collection
            gc.collect()

    async def generate_transcription_stream_from_data(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """
        Generate a Server-Sent-Events (SSE) transcription stream from prepared request data.

        Prepares the request for streaming, runs the model in a background thread, and yields SSE-formatted strings containing partial transcription chunks and timing metadata until completion. The function expects request_data to include an "audio_path" pointing to a saved temporary audio file; it will set request_data["stream"] = True and will remove the "audio_path" entry when invoking the model. After streaming completes (or on error), the temporary audio file is deleted and garbage collection is invoked.

        Parameters:
            request_data (dict[str, Any]): Prepared model request parameters. Must include "audio_path" (str). Other model options (e.g., temperature, language) may be present.

        Returns:
            AsyncGenerator[str, None]: An asynchronous generator that yields SSE data frames as strings. Each yielded string is a newline-terminated SSE event containing a JSON-encoded transcription chunk; after the final chunk a `"data: [DONE]\n\n"` frame is yielded.
        """
        request_id = f"transcription-{uuid.uuid4()}"
        created_time = int(time.time())
        temp_file_path = request_data.get("audio_path")

        try:
            # Set stream mode
            request_data["stream"] = True

            # Offload synchronous generator to thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            audio_path = request_data.pop("audio_path")

            def create_generator() -> Generator[dict[str, Any], None, None]:
                return cast(
                    "Generator[dict[str, Any], None, None]",
                    self.model(audio_path=audio_path, **request_data),
                )

            generator = await loop.run_in_executor(None, create_generator)

            try:
                while True:
                    chunk = await loop.run_in_executor(None, next, generator)
                    # Create streaming response
                    stream_response = TranscriptionResponseStream(
                        id=request_id,
                        object="transcription.chunk",
                        created=created_time,
                        model=self.model_path,
                        choices=[
                            TranscriptionResponseStreamChoice(
                                delta=Delta(
                                    content=chunk.get("text", ""),
                                    function_call=None,
                                    refusal=None,
                                    role=None,
                                    tool_calls=None,
                                    reasoning_content=None,
                                ),
                                finish_reason=None,
                            ),
                        ],
                    )

                    # Yield as SSE format
                    yield f"data: {stream_response.model_dump_json()}\n\n"
            except StopIteration:
                pass

            # Send final chunk with finish_reason
            final_response = TranscriptionResponseStream(
                id=request_id,
                object="transcription.chunk",
                created=created_time,
                model=self.model_path,
                choices=[
                    TranscriptionResponseStreamChoice(
                        delta=Delta(
                            content="",
                            function_call=None,
                            refusal=None,
                            role=None,
                            tool_calls=None,
                            reasoning_content=None,
                        ),
                        finish_reason="stop",
                    ),
                ],
            )
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during transcription streaming. {type(e).__name__}: {e}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")
            # Clean up
            gc.collect()

    async def _process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process an audio transcription request. This is the worker function for the request queue.

        Args:
            request_data: Dictionary containing the request data.

        Returns
        -------
            Dict: The model's response containing transcribed text.
        """
        try:
            # Extract request parameters
            audio_path = request_data.pop("audio_path")

            # Call the model with the audio file in a thread executor
            loop = asyncio.get_running_loop()

            def _run_model() -> dict[str, Any]:
                return cast(
                    "dict[str, Any]",
                    self.model(audio_path=audio_path, **request_data),
                )

            result: dict[str, Any] = await loop.run_in_executor(None, _run_model)

            # Force garbage collection after model inference
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing audio transcription request. {type(e).__name__}: {e}")
            # Clean up on error
            gc.collect()
            raise
        else:
            return result

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save the uploaded file to a temporary location.

        Args:
            file: The uploaded file object.

        Returns
        -------
            str: Path to the temporary file.
        """
        try:
            # Create a temporary file with the same extension as the uploaded file
            file_extension = Path(file.filename).suffix if file.filename else ".wav"

            logger.debug(f"file_extension: {file_extension}")

            # Read file content first (this can only be done once with FastAPI uploads)
            content = await file.read()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                # Write the file contents
                temp_file.write(content)
                temp_path = temp_file.name

            logger.debug(f"Saved uploaded file to temporary location: {temp_path}")

        except OSError as e:
            logger.error(f"Error saving uploaded file. ({type(e).__name__}): {e}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Internal server error while saving uploaded file",
            ) from e
        except Exception as e:
            logger.error(f"Error saving uploaded file. ({type(e).__name__}): {e}")
            raise
        else:
            return temp_path

    async def prepare_transcription_request(self, request: TranscriptionRequest) -> dict[str, Any]:
        """
        Builds a model-ready request dictionary and saves the uploaded audio to a temporary file.

        Saves the uploaded file, sets "audio_path" and "verbose" and includes any provided optional model parameters (e.g., temperature, language, initial_prompt, top_p, top_k, min_p, seed, frequency_penalty, repetition_penalty, presence_penalty).

        Parameters:
            request (TranscriptionRequest): Incoming transcription request containing the uploaded file and optional model parameters.

        Returns:
            dict: Request data ready for model inference, including "audio_path" and any provided parameters.

        Raises:
            HTTPException: If saving the uploaded file or preparing the request fails (returns a 400 Bad Request with a standardized error payload).
        """
        try:
            file = request.file

            file_path = await self._save_uploaded_file(file)
            request_data = {
                "audio_path": file_path,
                "verbose": False,
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                request_data["temperature"] = request.temperature

            if request.language is not None:
                request_data["language"] = request.language

            if request.prompt is not None:
                request_data["initial_prompt"] = request.prompt

            # Map additional tuning parameters if they exist
            if request.top_p is not None:
                request_data["top_p"] = request.top_p
            if request.top_k is not None:
                request_data["top_k"] = request.top_k
            if request.min_p is not None:
                request_data["min_p"] = request.min_p
            if request.seed is not None:
                request_data["seed"] = request.seed
            if request.frequency_penalty is not None:
                request_data["frequency_penalty"] = request.frequency_penalty
            if request.repetition_penalty is not None:
                request_data["repetition_penalty"] = request.repetition_penalty
            if request.presence_penalty is not None:
                request_data["presence_penalty"] = request.presence_penalty

            logger.debug(f"Prepared transcription request: {request_data}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare transcription request. {type(e).__name__}: {e}")
            content = create_error_response(
                f"Failed to process request: {e}",
                "bad_request",
                HTTPStatus.BAD_REQUEST,
            )
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=content) from e
        else:
            return request_data

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics from the request queue.

        Returns
        -------
            Dict with queue statistics.
        """
        return self.request_queue.get_queue_stats()

    async def cleanup(self) -> None:
        """
        Cleanup resources and stop the request queue before shutdown.

        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXWhisperHandler resources")
            if hasattr(self, "request_queue"):
                await self.request_queue.stop()
            logger.info("MLXWhisperHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXWhisperHandler cleanup. {type(e).__name__}: {e}")
