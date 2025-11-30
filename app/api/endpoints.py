"""API endpoints for the MLX OpenAI server."""

import base64
from collections.abc import AsyncGenerator
from http import HTTPStatus
import json
import random
import time
from typing import Annotated, Any, Literal, TypeAlias

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
import numpy as np

from ..handler import MFLUX_AVAILABLE, MLXFluxHandler
from ..handler.mlx_embeddings import MLXEmbeddingsHandler
from ..handler.mlx_lm import MLXLMHandler
from ..handler.mlx_vlm import MLXVLMHandler
from ..handler.mlx_whisper import MLXWhisperHandler
from ..schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    Delta,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    FunctionCall,
    HealthCheckResponse,
    HealthCheckStatus,
    ImageEditRequest,
    ImageEditResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    Model,
    ModelsResponse,
    StreamingChoice,
    TranscriptionRequest,
    TranscriptionResponse,
    UsageInfo,
)
from ..utils.errors import create_error_response
from .hub_routes import (
    HubConfigError,
    _load_hub_config_from_request,
    get_cached_model_metadata,
    get_configured_model_id,
    get_running_hub_models,
    hub_load_model,
    hub_router,
    hub_start_model,
    hub_status,
    hub_status_page,
    hub_stop_model,
    hub_unload_model,
)

router = APIRouter()
router.include_router(hub_router)

__all__ = [
    "hub_load_model",
    "hub_start_model",
    "hub_status",
    "hub_status_page",
    "hub_stop_model",
    "hub_unload_model",
    "router",
]


MLXHandlerType: TypeAlias = (
    MLXVLMHandler | MLXLMHandler | MLXFluxHandler | MLXEmbeddingsHandler | MLXWhisperHandler
)


async def _get_handler_or_error(
    raw_request: Request,
    reason: str,
    *,
    model_name: str | None = None,
) -> tuple[MLXHandlerType | None, JSONResponse | None]:
    """Return a loaded handler or an error response if unavailable.

    Parameters
    ----------
    raw_request : Request
        Incoming FastAPI request.
    reason : str
        Context string used for logging and load tracking.
    model_name : str | None, optional
        Explicit model identifier supplied by the caller.

    Returns
    -------
    tuple[MLXHandlerType | None, JSONResponse | None]
        A handler/error tuple where exactly one entry is ``None``.
    """
    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is not None and model_name is not None:
        target = model_name.strip()
        if not target:
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        "Model name is required when running the hub server",
                        "invalid_request_error",
                        HTTPStatus.BAD_REQUEST,
                    ),
                    status_code=HTTPStatus.BAD_REQUEST,
                ),
            )
        try:
            handler = await controller.acquire_handler(target, reason=reason)
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception(
                f"Unable to load handler for model '{target}'. {type(e).__name__}: {e}",
            )
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        f"Failed to load model '{target}' handler",
                        "server_error",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
            )
        else:
            return handler, None

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    if handler_manager is not None:
        try:
            handler = await handler_manager.ensure_loaded(reason)
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception(
                f"Unable to load handler via JIT for {reason}. {type(e).__name__}: {e}",
            )
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        "Failed to load model handler",
                        "server_error",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
            )

        if handler is not None:
            return handler, None

    handler = getattr(raw_request.app.state, "handler", None)
    if handler is None:
        return (
            None,
            JSONResponse(
                content=create_error_response(
                    "Model handler not initialized",
                    "service_unavailable",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ),
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            ),
        )
    return handler, None


def _hub_model_required_error() -> JSONResponse:
    """
    Return a 400 JSONResponse indicating a model selection is required when the server runs in hub mode.

    The response payload is an OpenAI-style error explaining that the 'model' field (or query parameter) must be provided and suggests calling /v1/models to list available names.

    Returns:
        JSONResponse: A response containing the error payload with HTTP status code 400.
    """
    return JSONResponse(
        content=create_error_response(
            "Model selection is required when the server runs in hub mode. "
            "Include the 'model' field (or query parameter) to choose a registered model. "
            "Call /v1/models to list available names.",
            "invalid_request_error",
            HTTPStatus.BAD_REQUEST,
        ),
        status_code=HTTPStatus.BAD_REQUEST,
    )


def _resolve_model_for_openai_api(
    raw_request: Request,
    model_name: str | None,
    *,
    provided_explicitly: bool,
) -> tuple[str | None, str | None, JSONResponse | None]:
    """
    Resolve which internal hub handler and external model identifier should be used for an OpenAI-style request.

    Parameters:
        raw_request (Request): FastAPI request object used to access app state (hub controller and config).
        model_name (str | None): The OpenAI-style model identifier provided by the client (may be None).
        provided_explicitly (bool): Whether the client explicitly included a `model` field in the request payload.

    Returns:
        tuple[api_model_id, handler_name, error_response]:
            api_model_id (str | None): The model identifier to expose through the OpenAI-compatible API (typically the registry `model_path`), or None if resolution failed.
            handler_name (str | None): The hub controller name used to acquire a handler, or None if resolution failed.
            error_response (JSONResponse | None): A JSONResponse describing the error when resolution or validation fails (e.g., hub mode requires an explicit model or the requested model is not started); None on success.
    """
    normalized = (model_name or "").strip() or None
    controller = getattr(raw_request.app.state, "hub_controller", None)
    # Non-hub mode: handler name and API id are the same
    if controller is None:
        return normalized, normalized, None

    # Hub mode requires explicit model selection
    if not provided_explicitly or normalized is None:
        return None, None, _hub_model_required_error()

    # Try to load hub config to map between name and model_path
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError:
        config = None

    mapped_handler: str | None = None
    mapped_api_id: str | None = None

    if config is not None:
        for m in getattr(config, "models", []):
            # m.name may be None; use model_identifier as fallback
            cfg_name = getattr(m, "name", None) or getattr(m, "model_identifier", None)
            cfg_path = getattr(m, "model_path", None)
            if normalized in {cfg_name, cfg_path, getattr(m, "model_identifier", None)}:
                mapped_handler = cfg_name
                mapped_api_id = cfg_path
                break

    # If we didn't map from config, assume the provided value might be a handler name
    if mapped_handler is None:
        mapped_handler = normalized
    if mapped_api_id is None:
        # If no mapping available, expose the normalized value as the API id
        mapped_api_id = normalized

    # Validate that the handler is running if we can query running models
    running_models = get_running_hub_models(raw_request)
    if running_models is not None:
        # running_models contains hub names; ensure mapped_handler is started
        if mapped_handler not in running_models:
            return (
                None,
                None,
                JSONResponse(
                    content=create_error_response(
                        f"Model '{mapped_api_id}' is not started. Start the process before sending requests.",
                        "invalid_request_error",
                        HTTPStatus.BAD_REQUEST,
                    ),
                    status_code=HTTPStatus.BAD_REQUEST,
                ),
            )

    return mapped_api_id, mapped_handler, None


def _model_field_was_provided(payload: Any) -> bool:
    """
    Detect whether the payload explicitly included a "model" field.

    Parameters:
        payload (Any): The request payload (typically a Pydantic model) to inspect.

    Returns:
        bool: `True` if the payload's field-set indicates "model" was provided, `False` otherwise.
    """
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", None)
    if not isinstance(fields_set, set):
        return False
    return "model" in fields_set


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health", response_model=None)
async def health(raw_request: Request) -> HealthCheckResponse | JSONResponse:
    """
    Return current service health and model status based on app state (handler manager, hub controller, or direct handler).

    Parameters:
        raw_request (Request): Incoming FastAPI request used to inspect application state (handler_manager, hub_controller, or handler) and to determine the configured model id.

    Returns:
        HealthCheckResponse: When the service can report status, contains `status` (OK), `model_id`, and `model_status` ("initialized", "unloaded", or "controller").
        JSONResponse: When no handler is initialized, returns a 503 response with content {"status": "unhealthy", "model_id": None, "model_status": "uninitialized"}.
    """
    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    configured_model_id = get_configured_model_id(raw_request)
    controller = getattr(raw_request.app.state, "hub_controller", None)

    if handler_manager is not None:
        handler = getattr(handler_manager, "current_handler", None)
        if handler is not None:
            model_id = getattr(handler, "model_path", configured_model_id or "unknown")
            return HealthCheckResponse(
                status=HealthCheckStatus.OK,
                model_id=model_id,
                model_status="initialized",
            )
        return HealthCheckResponse(
            status=HealthCheckStatus.OK,
            model_id=configured_model_id,
            model_status="unloaded",
        )

    if controller is not None:
        return HealthCheckResponse(
            status=HealthCheckStatus.OK,
            model_id=configured_model_id,
            model_status="controller",
        )

    handler = getattr(raw_request.app.state, "handler", None)
    if handler is None:
        # Handler not initialized - return 503 with degraded status
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "model_id": None, "model_status": "uninitialized"},
        )

    model_id = getattr(handler, "model_path", configured_model_id or "unknown")
    return HealthCheckResponse(
        status=HealthCheckStatus.OK,
        model_id=model_id,
        model_status="initialized",
    )


@router.get("/v1/models", response_model=None)
async def models(raw_request: Request) -> ModelsResponse | JSONResponse:
    """
    Return the list of models currently available to the server.

    The response is resolved from the highest-priority source available: the model registry (filtered to running hub models when applicable), a cached metadata snapshot, or the active handler as a backward-compatible fallback. On failure returns a JSONResponse containing an OpenAI-style error.

    Parameters:
        raw_request (Request): The incoming FastAPI request; used to access app state (registry, supervisor, cache, handlers).

    Returns:
        ModelsResponse or JSONResponse: A ModelsResponse with object="list" and model entries when successful, or a JSONResponse with an error payload on failure.
    """
    # Try registry first (Phase 1+), fall back to handler for backward compat
    registry = getattr(raw_request.app.state, "model_registry", None)
    supervisor = getattr(raw_request.app.state, "supervisor", None)
    if registry is not None:
        try:
            running_models: list[str] | None = None
            supervisor_status = None
            if supervisor is not None:
                # Hub daemon: get running models directly from supervisor
                supervisor_status = supervisor.get_status()
                running_models = [
                    model["model_path"]
                    for model in supervisor_status.get("models", [])
                    if model.get("state") == "running" and model.get("model_path")
                ]
            else:
                # Separate server: get running models from hub daemon
                temp = get_running_hub_models(raw_request)
                running_models = list(temp) if temp is not None else None

            if running_models is not None:
                models_data = registry.list_models()
                # Update vram_loaded from supervisor status if available
                if supervisor_status is not None:
                    memory_lookup = {
                        model["model_path"]: model.get("memory_loaded", False)
                        for model in supervisor_status.get("models", [])
                        if model.get("model_path")
                    }
                    for model in models_data:
                        model_path = model.get("id")
                        if model_path in memory_lookup:
                            model.setdefault("metadata", {})["vram_loaded"] = memory_lookup[
                                model_path
                            ]
                models_data = [model for model in models_data if model.get("id") in running_models]
                return ModelsResponse(object="list", data=[Model(**model) for model in models_data])
            # If running_models is None, fall through to handler fallback
        except Exception as e:
            logger.error(f"Error retrieving models from registry. {type(e).__name__}: {e}")
            return JSONResponse(
                content=create_error_response(
                    f"Failed to retrieve models: {e}",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    cached_metadata = get_cached_model_metadata(raw_request)
    if cached_metadata is not None:
        return ModelsResponse(object="list", data=[Model(**cached_metadata)])

    # Fallback to handler (Phase 0 compatibility)
    handler, error = await _get_handler_or_error(raw_request, "models")
    if error is not None:
        return error
    if handler is None:
        # No handler available (e.g., hub mode with no running hub), return empty list
        return ModelsResponse(object="list", data=[])

    try:
        models_data = await handler.get_models()
        return ModelsResponse(object="list", data=[Model(**model) for model in models_data])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving models. {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(
                f"Failed to retrieve models: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.get("/v1/queue/stats", response_model=None)
async def queue_stats(
    raw_request: Request,
    model: str | None = Query(None, description="Optional model name"),
) -> dict[str, Any] | JSONResponse:
    """
    Return queue statistics for the service or a specific model.

    Resolves the OpenAI-style model identifier to an internal handler (respecting hub mode rules) and queries that handler for queue statistics. The shape and keys of `queue_stats` depend on the handler implementation (e.g., Flux vs LM/VLM/Whisper). If hub mode requires an explicit model or the handler cannot be loaded, a JSONResponse with an appropriate error is returned.

    Parameters:
        raw_request (Request): Incoming FastAPI request used to resolve hub/handler context.
        model (str | None): Optional API model identifier to filter statistics; may be required when the server is running in hub mode.

    Returns:
        dict: A dictionary with "status": "ok" and "queue_stats" containing handler-dependent statistics, or a JSONResponse containing an error description.
    """
    _api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, model, provided_explicitly=model is not None
    )
    if validation_error is not None:
        return validation_error

    handler, error = await _get_handler_or_error(
        raw_request,
        "queue_stats",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    try:
        stats = await handler.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return JSONResponse(
            content=create_error_response(
                "Failed to get queue stats",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    else:
        return {"status": "ok", "queue_stats": stats}


# =============================================================================
# API Endpoints - Core functionality
# =============================================================================


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """
    Handle an OpenAI-style chat completion request and return either a final response, a streaming response, or an error response.

    Resolves the requested model to an internal handler, validates that the handler supports chat (text or multimodal), and delegates to the appropriate processing routine which yields either a ChatCompletionResponse or a StreamingResponse. On failure returns a JSONResponse containing an OpenAI-compatible error.

    Parameters:
        request (ChatCompletionRequest): The OpenAI-style chat completion payload; may include an explicit `model` field.
        raw_request (Request): The incoming FastAPI request (used for routing context and optional request_id).

    Returns:
        ChatCompletionResponse | StreamingResponse | JSONResponse: A completed chat response, a streaming SSE response for streamed requests, or a JSONResponse containing an error description.
    """
    api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, request.model, provided_explicitly=_model_field_was_provided(request)
    )
    if validation_error is not None:
        return validation_error
    # Expose model_path (api_model_id) in OpenAI-compatible responses
    if api_model_id is not None:
        request.model = api_model_id

    handler, error = await _get_handler_or_error(
        raw_request,
        "chat_completions",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    request_id = getattr(raw_request.state, "request_id", None)

    try:
        if isinstance(handler, MLXVLMHandler):
            return await process_multimodal_request(handler, request, request_id)
        return await process_text_request(handler, request, request_id)
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(f"Error processing chat completion request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.post("/v1/embeddings", response_model=None)
async def embeddings(
    request: EmbeddingRequest,
    raw_request: Request,
) -> EmbeddingResponse | JSONResponse:
    """
    Handle an OpenAI-style embeddings request by resolving the target model, invoking the embeddings handler, and returning the embeddings or an error response.

    Parameters:
        request (EmbeddingRequest): OpenAI-compatible embeddings payload; the `model` field may be remapped to an internal API model id.
        raw_request (Request): The incoming FastAPI request used for routing and handler resolution.

    Returns:
        EmbeddingResponse: Embeddings formatted according to `request.encoding_format`.
        JSONResponse: OpenAI-compatible error response when model resolution, handler selection, or processing fails.
    """
    api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, request.model, provided_explicitly=_model_field_was_provided(request)
    )
    if validation_error is not None:
        return validation_error
    if api_model_id is not None:
        request.model = api_model_id

    handler, error = await _get_handler_or_error(
        raw_request,
        "embeddings",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXEmbeddingsHandler):
        return JSONResponse(
            content=create_error_response(
                "Embedding requests require an embeddings model. Use --model-type embeddings.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        embeddings_response = await handler.generate_embeddings_response(request)
        return create_response_embeddings(
            embeddings_response,
            request.model,
            request.encoding_format,
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(f"Error processing embedding request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.post("/v1/images/generations", response_model=None)
async def image_generations(
    request: ImageGenerationRequest,
    raw_request: Request,
) -> ImageGenerationResponse | JSONResponse:
    """
    Process an OpenAI-style image generation request, resolve the target model, and delegate to an image-generation-capable handler.

    Resolves OpenAI-style model identifiers to an internal handler (honoring hub mode), ensures the resolved handler supports image generation, and invokes the handler to produce an ImageGenerationResponse. Returns a JSONResponse containing a standardized error when model resolution fails, no handler is available, the selected handler is not an image-generation type, or an internal error occurs.

    Parameters:
        request (ImageGenerationRequest): The image generation request payload; `model` may be rewritten to the resolved API-visible model id.
        raw_request (Request): The incoming FastAPI request used for hub/handler resolution.

    Returns:
        ImageGenerationResponse or JSONResponse: The successful image generation result, or a JSONResponse containing an OpenAI-compatible error payload.
    """
    api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, request.model, provided_explicitly=_model_field_was_provided(request)
    )
    if validation_error is not None:
        return validation_error
    if api_model_id is not None:
        request.model = api_model_id

    handler, error = await _get_handler_or_error(
        raw_request,
        "image_generation",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image generation requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        image_response: ImageGenerationResponse = await handler.generate_image(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image generation request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    else:
        return image_response


@router.post("/v1/images/edits", response_model=None)
async def create_image_edit(
    request: Annotated[ImageEditRequest, Form()],
    raw_request: Request,
) -> ImageEditResponse | JSONResponse:
    """
    Handle an image edit request, routing to a per-model handler and returning the edited image or an error response.

    Parameters:
        request (ImageEditRequest): The OpenAI-style image edit payload (may include an explicit `model` field).
        raw_request (Request): The incoming FastAPI request used to resolve hub/handler context.

    Returns:
        ImageEditResponse: Successful image edit result.
        JSONResponse: Error response when model resolution, handler availability, handler type, or processing fails.

    Notes:
        - Re-raises HTTPException produced by handlers.
        - Returns 400 if the selected handler does not support image generation, 503 if no handler is initialized, and 500 for unexpected processing errors.
    """
    api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, request.model, provided_explicitly=_model_field_was_provided(request)
    )
    if validation_error is not None:
        return validation_error
    if api_model_id is not None:
        request.model = api_model_id

    handler, error = await _get_handler_or_error(
        raw_request,
        "image_edit",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image editing requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    try:
        image_response: ImageEditResponse = await handler.edit_image(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image edit request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    else:
        return image_response


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_audio_transcriptions(
    request: Annotated[TranscriptionRequest, Form()],
    raw_request: Request,
) -> StreamingResponse | TranscriptionResponse | JSONResponse | str:
    """
    Handle an audio transcription request using the configured Whisper handler.

    When a hub model mapping is present, the request's model field may be replaced with the resolved API model id. If the request requests streaming, returns a Server-Sent Events (SSE) stream; otherwise returns the final transcription.

    Parameters:
        request (TranscriptionRequest): The transcription request payload; may be modified with a resolved API model id.
        raw_request (Request): The incoming FastAPI request (used for routing and handler resolution).

    Returns:
        StreamingResponse: An SSE stream of transcription events when `request.stream` is true.
        TranscriptionResponse or str: The completed transcription result when not streaming.
        JSONResponse: An error response when model resolution, handler selection, or processing fails.
    """
    api_model_id, handler_name, validation_error = _resolve_model_for_openai_api(
        raw_request, request.model, provided_explicitly=_model_field_was_provided(request)
    )
    if validation_error is not None:
        return validation_error
    if api_model_id is not None:
        request.model = api_model_id

    handler, error = await _get_handler_or_error(
        raw_request,
        "audio_transcriptions",
        model_name=handler_name,
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXWhisperHandler):
        return JSONResponse(
            content=create_error_response(
                "Audio transcription requests require a whisper model. Use --model-type whisper.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        if request.stream:
            # process the request before sending to the handler
            request_data = await handler.prepare_transcription_request(request)
            return StreamingResponse(
                handler.generate_transcription_stream_from_data(request_data),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        transcription_response: (
            TranscriptionResponse | str
        ) = await handler.generate_transcription_response(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing transcription request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    else:
        return transcription_response


def create_response_embeddings(
    embeddings: list[list[float]],
    model: str,
    encoding_format: Literal["float", "base64"] = "float",
) -> EmbeddingResponse:
    """
    Create an OpenAI-style EmbeddingResponse from a list of embedding vectors.

    When `encoding_format` is "base64", each embedding is serialized as float32 bytes and base64-encoded; when "float", embeddings are returned as lists of floats.

    Parameters:
        embeddings (list[list[float]]): Embedding vectors to include in the response.
        model (str): Model identifier to set on the response.
        encoding_format (Literal["float", "base64"], optional): Encoding for each embedding; defaults to "float".

    Returns:
        EmbeddingResponse: Response with `object="list"`, `data` containing EmbeddingResponseData entries, and `model` set to the provided model.
    """
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        if encoding_format == "base64":
            # Convert list/array to bytes before base64 encoding
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            embeddings_response.append(
                EmbeddingResponseData(
                    embedding=base64.b64encode(embedding_bytes).decode("utf-8"),
                    index=index,
                ),
            )
        else:
            embeddings_response.append(EmbeddingResponseData(embedding=embedding, index=index))
    return EmbeddingResponse(object="list", data=embeddings_response, model=model, usage=None)


def create_response_chunk(
    chunk: str | dict[str, Any],
    model: str,
    *,
    is_final: bool = False,
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    | None = "stop",
    chat_id: str | None = None,
    created_time: int | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
) -> ChatCompletionChunk:
    """
    Builds a ChatCompletionChunk for OpenAI-compatible streaming from a text, reasoning, or tool-call chunk.

    The function accepts either a string or a dict-shaped chunk and produces a ChatCompletionChunk whose delta encodes one of:
    - text content,
    - reasoning content (optionally with content),
    - a tool/function call (with `name` and/or `arguments`).

    Parameters:
        chunk (str | dict[str, Any]): The incoming chunk to format. If a string, it is treated as assistant content. If a dict, expected keys include `content`, `reasoning_content`, `name`, `arguments`, and optional `index`.
        model (str): Model identifier to include in the chunk.
        is_final (bool, optional): If True, the choice's finish_reason is set to `finish_reason`. Default False.
        finish_reason (Literal["stop","length","tool_calls","content_filter","function_call"] | None, optional):
            The finish reason to attach when `is_final` is True.
        chat_id (str | None, optional): ID for the chat completion chunk. If omitted, a new id is generated.
        created_time (int | None, optional): Unix timestamp to set as the chunk's `created`. If omitted, the current time is used.
        request_id (str | None, optional): Optional request identifier to propagate into the chunk.
        tool_call_id (str | None, optional): Identifier to use for tool/function call chunks; if omitted a new tool-call id is generated when needed.

    Returns:
        ChatCompletionChunk: A prepared chat completion chunk whose delta contains the appropriate fields for streaming (content, reasoning_content, or tool call) and whose finish_reason is set when `is_final` is True.
    """
    chat_id = chat_id or get_id()
    created_time = created_time or int(time.time())

    # Handle string chunks (text content)
    if isinstance(chunk, str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(content=chunk, role="assistant"),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,
                ),
            ],
            request_id=request_id,
        )

    # Handle reasoning content chunks
    if "reasoning_content" in chunk:
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(
                        reasoning_content=chunk["reasoning_content"],
                        role="assistant",
                        content=chunk.get("content", None),
                    ),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,
                ),
            ],
            request_id=request_id,
        )

    # Handle dict chunks with only content (no reasoning or tool calls)
    if "content" in chunk and isinstance(chunk["content"], str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(content=chunk["content"], role="assistant"),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,
                ),
            ],
            request_id=request_id,
        )

    # Handle tool/function call chunks
    function_call = None
    if "name" in chunk:
        function_call = ChoiceDeltaFunctionCall(name=chunk["name"], arguments=None)
        if "arguments" in chunk:
            function_call.arguments = chunk["arguments"]
    elif "arguments" in chunk:
        # Handle case where arguments come before name (streaming)
        function_call = ChoiceDeltaFunctionCall(name=None, arguments=chunk["arguments"])

    if function_call:
        # Validate index exists before accessing
        tool_index = chunk.get("index", 0)
        tool_identifier = tool_call_id or get_tool_call_id()
        tool_chunk = ChoiceDeltaToolCall(
            index=tool_index,
            type="function",
            id=tool_identifier,
            function=function_call,
        )

        delta = Delta(content=None, role="assistant", tool_calls=[tool_chunk])  # type: ignore[call-arg]
    else:
        # Fallback: create empty delta if no recognized chunk type
        delta = Delta(role="assistant")  # type: ignore[call-arg]

    return ChatCompletionChunk(
        id=chat_id,
        object="chat.completion.chunk",
        created=created_time,
        model=model,
        choices=[
            StreamingChoice(
                index=0, delta=delta, finish_reason=finish_reason if is_final else None
            ),
        ],
        request_id=request_id,
    )


def _yield_sse_chunk(data: dict[str, Any] | ChatCompletionChunk) -> str:
    """
    Format a value as a Server-Sent Event (SSE) data string.

    If `data` is a ChatCompletionChunk it will be converted via its `model_dump()` method; otherwise the value is JSON-serialized directly. The result is prefixed with "data: " and terminated with a double newline to form a single SSE event.

    Parameters:
        data (dict[str, Any] | ChatCompletionChunk): Payload to encode as the SSE event data.

    Returns:
        str: SSE-formatted string like `data: <json>\n\n`.
    """
    if isinstance(data, ChatCompletionChunk):
        return f"data: {json.dumps(data.model_dump())}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def handle_stream_response(
    generator: AsyncGenerator[Any, None],
    model: str,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream OpenAI-compatible Server-Sent Events (SSE) from an async chunk generator.

    This function consumes an async generator that yields streaming response pieces (strings or dicts), formats them into OpenAI-compatible chat completion SSE chunks, and yields the serialized SSE payloads. It emits an initial role-only assistant delta, intermediate content/tool-call/usage chunks, a final chunk that includes the finish reason and any collected usage, and the explicit "[DONE]" sentinel. On exceptions it yields an error chunk before the finalization.

    Parameters:
        generator (AsyncGenerator[Any, None]): Async generator that yields streaming pieces. Accepted chunk shapes:
            - str: content delta to append to the assistant message.
            - dict: structured chunk that may contain tool call fields (`name`, `arguments`, optional `index`) or a special `__usage__` key to supply usage metadata.
        model (str): Model identifier to include in emitted chunks.
        request_id (str | None): Optional request identifier to include in emitted chunks.

    Returns:
        AsyncGenerator[str, None]: SSE-formatted strings representing serialized ChatCompletion chunks and the final "data: [DONE]" sentinel.
    """
    chat_index = get_id()
    created_time = int(time.time())
    finish_reason = "stop"
    next_tool_call_index = 0
    current_implicit_tool_index: int | None = None
    tool_call_ids: dict[int, str] = {}
    usage_info = None

    try:
        # First chunk: role-only delta, as per OpenAI
        first_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"))],  # type: ignore[call-arg]
            request_id=request_id,
        )
        yield _yield_sse_chunk(first_chunk)

        async for chunk in generator:
            if not chunk:
                continue

            if isinstance(chunk, str):
                response_chunk = create_response_chunk(
                    chunk,
                    model,
                    chat_id=chat_index,
                    created_time=created_time,
                    request_id=request_id,
                )
                yield _yield_sse_chunk(response_chunk)

            elif isinstance(chunk, dict):
                # Check if this is usage info from the handler
                if "__usage__" in chunk:
                    usage_info = chunk["__usage__"]
                    continue

                # Handle tool call chunks
                payload = dict(chunk)  # Create a copy to avoid mutating the original
                current_tool_id = None

                has_name = bool(payload.get("name"))
                has_arguments = "arguments" in payload
                payload_index = payload.get("index")

                if has_name:
                    finish_reason = "tool_calls"
                    if payload_index is None:
                        if current_implicit_tool_index is not None:
                            payload_index = current_implicit_tool_index
                        else:
                            payload_index = next_tool_call_index
                            next_tool_call_index += 1
                        payload["index"] = payload_index
                    current_implicit_tool_index = payload_index
                    # Keep the implicit index available for additional argument chunks
                elif has_arguments:
                    if payload_index is None:
                        if current_implicit_tool_index is not None:
                            payload_index = current_implicit_tool_index
                        else:
                            payload_index = next_tool_call_index
                            next_tool_call_index += 1
                        payload["index"] = payload_index
                    current_implicit_tool_index = payload_index
                elif payload_index is not None:
                    current_implicit_tool_index = payload_index

                payload_index = payload.get("index")
                if payload_index is not None:
                    if payload_index not in tool_call_ids:
                        tool_call_ids[payload_index] = get_tool_call_id()
                    current_tool_id = tool_call_ids[payload_index]

                response_chunk = create_response_chunk(
                    payload,
                    model,
                    chat_id=chat_index,
                    created_time=created_time,
                    request_id=request_id,
                    tool_call_id=current_tool_id,
                )
                yield _yield_sse_chunk(response_chunk)

            else:
                error_response = create_error_response(
                    f"Invalid chunk type: {type(chunk)}",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                yield _yield_sse_chunk(error_response)

    except HTTPException as e:
        logger.exception(f"HTTPException in stream wrapper: {type(e).__name__}: {e}")
        detail = e.detail if isinstance(e.detail, dict) else {"message": str(e)}
        error_response = detail  # type: ignore[assignment]
        yield _yield_sse_chunk(error_response)
    except Exception as e:
        logger.exception(f"Error in stream wrapper: {type(e).__name__}: {e}")
        error_response = create_error_response(
            str(e),
            "server_error",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        yield _yield_sse_chunk(error_response)
    finally:
        # Final chunk: finish_reason with usage info, as per OpenAI
        final_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(), finish_reason=finish_reason)],  # type: ignore[call-arg,arg-type]
            usage=usage_info,
            request_id=request_id,
        )
        yield _yield_sse_chunk(final_chunk)
        yield "data: [DONE]\n\n"


async def process_multimodal_request(
    handler: MLXVLMHandler,
    request: ChatCompletionRequest,
    request_id: str | None = None,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """
    Handle a multimodal chat completion request and return either a streamed SSE response or a final chat completion response.

    If the request requests streaming, produce a Server-Sent Events streaming response; otherwise obtain the handler's multimodal result and format it into an OpenAI-compatible chat completion response. When the handler returns a dict containing "response" and "usage", the usage is applied to the formatted response.

    Parameters:
        handler (MLXVLMHandler): Multimodal handler used to generate the response.
        request (ChatCompletionRequest): OpenAI-style chat completion request payload.
        request_id (str | None): Optional request identifier for tracing/logging.

    Returns:
        ChatCompletionResponse or StreamingResponse or JSONResponse: A finalized chat completion response, a streaming SSE response, or an error JSON response.
    """
    if request_id:
        logger.info(f"Processing multimodal request [request_id={request_id}]")

    if request.stream:
        return StreamingResponse(
            handle_stream_response(
                handler.generate_multimodal_stream(request),
                request.model,
                request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    # Extract response and usage from handler
    result = await handler.generate_multimodal_response(request)
    if isinstance(result, dict) and "response" in result and "usage" in result:
        response_data = result.get("response")
        usage = result.get("usage")
        return format_final_response(response_data, request.model, request_id, usage)  # type: ignore[arg-type]

    # Fallback for backward compatibility or if structure is different
    return format_final_response(result, request.model, request_id)


async def process_text_request(
    handler: MLXLMHandler | MLXVLMHandler,
    request: ChatCompletionRequest,
    request_id: str | None = None,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """
    Handle a text-only chat completion request, returning either a streamed SSE response or a completed chat response.

    Parameters:
        handler (MLXLMHandler | MLXVLMHandler): Model handler capable of producing text responses or text streams.
        request (ChatCompletionRequest): OpenAI-style chat completion request; its `stream` field controls streaming vs final response.
        request_id (str | None): Optional request identifier for logging and correlation.

    Returns:
        ChatCompletionResponse if the request is non-streaming; StreamingResponse (text/event-stream) that yields SSE-formatted chunks if `request.stream` is true; JSONResponse on error.
    """
    if request_id:
        logger.info(f"Processing text request [request_id={request_id}]")

    if request.stream:
        return StreamingResponse(
            handle_stream_response(
                handler.generate_text_stream(request),  # type: ignore[union-attr]
                request.model,
                request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Extract response and usage from handler
    result = await handler.generate_text_response(request)  # type: ignore[union-attr]
    response_data = result.get("response")
    usage = result.get("usage")
    return format_final_response(response_data, request.model, request_id, usage)  # type: ignore[arg-type]


def get_id() -> str:
    """
    Create a unique chat completion identifier.

    Returns:
        str: Identifier in the format "chatcmpl_<unix_timestamp><6-digit_random>", e.g. "chatcmpl_1700000000123456".
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"


def get_tool_call_id() -> str:
    """
    Create a unique identifier for a tool call.

    The ID uses the current Unix timestamp and a six-digit random suffix, formatted as:
    `call_<timestamp><6-digit-random>`.

    Returns:
        Identifier string for the tool call.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"


def format_final_response(
    response: str | dict[str, Any],
    model: str,
    request_id: str | None = None,
    usage: UsageInfo | None = None,
) -> ChatCompletionResponse:
    """
    Builds an OpenAI-compatible ChatCompletionResponse from a final model response.

    If `response` is a string, the result contains a single assistant message with that text and a finish reason of "stop". If `response` is a dict it may include the keys:
    - "content": assistant message content,
    - "reasoning_content": optional auxiliary reasoning text,
    - "tool_calls": optional list of tool call objects.

    Each entry in "tool_calls" is converted into a ChatCompletionMessageToolCall with a generated tool-call id; its "arguments" field is used as-is if already a string, otherwise it is serialized to JSON. When tool calls are present the response's finish reason is "tool_calls"; otherwise it is "stop".

    Parameters:
        response (str | dict[str, Any]): Final model response (string or structured dict).
        model (str): Model identifier to include in the response.
        request_id (str | None): Optional request id to attach to the response.
        usage (UsageInfo | None): Optional usage information to attach to the response.

    Returns:
        ChatCompletionResponse: Formatted chat completion response matching OpenAI-style schema.
    """
    if isinstance(response, str):
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response,
                        refusal=None,
                        reasoning_content=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=usage,
            request_id=request_id,
        )

    reasoning_content = response.get("reasoning_content", None)
    response_content = response.get("content", None)
    tool_calls = response.get("tool_calls", None)
    tool_call_responses = []
    if tool_calls is None or len(tool_calls) == 0:
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response_content,
                        reasoning_content=reasoning_content,
                        refusal=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=usage,
            request_id=request_id,
        )
    for idx, tool_call in enumerate(tool_calls):
        arguments = tool_call.get("arguments")
        # If arguments is already a string, use it directly; otherwise serialize it
        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments)
        function_call = FunctionCall(name=tool_call.get("name"), arguments=arguments_str)
        tool_call_response = ChatCompletionMessageToolCall(
            id=get_tool_call_id(),
            type="function",
            function=function_call,
            index=idx,
        )
        tool_call_responses.append(tool_call_response)

    message = Message(
        role="assistant",
        content=response_content,
        reasoning_content=reasoning_content,
        tool_calls=tool_call_responses,
        refusal=None,
        tool_call_id=None,
    )

    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        usage=usage,
        request_id=request_id,
    )
