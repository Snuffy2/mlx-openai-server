"""Ollama-compatible API endpoints.

This module exposes a subset of the Ollama REST API on top of the
existing MLX handlers so that both the OpenAI and Ollama protocols can
run side-by-side. Wherever possible, it reuses the OpenAI schemas and
handler helpers to avoid code duplication and to guarantee identical
model behavior across surfaces.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from http import HTTPStatus
import json
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from ..handler import MLXEmbeddingsHandler, MLXLMHandler, MLXVLMHandler
from ..schemas.openai import (
    ChatCompletionContentPartImage,
    ChatCompletionContentPartText,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    ImageURL,
    OpenAIBaseModel,
    UsageInfo,
)
from ..utils.errors import create_error_response
from ..version import __version__
from .endpoints import MLXHandlerType, _get_handler_or_error, format_final_response

router = APIRouter(prefix="/api", tags=["ollama"])

NDJSON_MEDIA_TYPE = "application/x-ndjson"
NDJSON_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
ResponseKind = Literal["chat", "generate"]
OPTION_FIELD_MAP: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "seed": "seed",
    "num_predict": "max_tokens",
    "stop": "stop",
    "repeat_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


class OllamaGenerateRequest(OpenAIBaseModel):
    """Request model for /api/generate."""

    model: str
    prompt: str | None = None
    suffix: str | None = None
    system: str | None = None
    stream: bool = True
    images: list[str] | None = None
    options: dict[str, Any] | None = None
    format: str | dict[str, Any] | None = None
    keep_alive: str | int | None = None

    def to_chat_request(self) -> ChatCompletionRequest:
        """Convert a generate payload into a ChatCompletionRequest."""

        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})

        if self.images:
            parts: list[dict[str, Any]] = []
            if self.prompt:
                parts.append(
                    ChatCompletionContentPartText(text=self.prompt, type="text").model_dump()
                )
            parts.extend(
                ChatCompletionContentPartImage(
                    image_url=ImageURL(url=image),
                    type="image_url",
                ).model_dump()
                for image in self.images
            )
            user_content: str | list[dict[str, Any]] = parts
        else:
            user_content = self.prompt or ""

        messages.append({"role": "user", "content": user_content})
        chat_request = ChatCompletionRequest.model_validate(
            {"model": self.model, "messages": messages, "stream": self.stream}
        )

        response_format = _prepare_response_format(self.format)
        if response_format is not None:
            chat_request = chat_request.model_copy(update={"response_format": response_format})

        if self.suffix:
            logger.warning("Suffix parameter for /api/generate is not currently supported")

        return _apply_ollama_options(chat_request, self.options)


class LifecycleRequest(OpenAIBaseModel):
    """Simple body schema for load/unload endpoints."""

    model: str | None = None
    keep_alive: str | int | None = None


def _iso_timestamp(epoch_seconds: int | None = None) -> str:
    """Convert epoch seconds into an ISO-8601 timestamp.

    Parameters
    ----------
    epoch_seconds : int | None, optional
        Epoch timestamp in seconds. When ``None`` the current time is used.

    Returns
    -------
    str
        UTC timestamp formatted without timezone offset.
    """

    ts = (
        datetime.fromtimestamp(epoch_seconds, UTC)
        if epoch_seconds is not None
        else datetime.now(UTC)
    )
    return ts.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_message_content(content: str | list[Any] | None) -> str:
    """Flatten a chat message's content into a single string.

    Parameters
    ----------
    content : str | list[Any] | None
        Raw content from an OpenAI message payload.

    Returns
    -------
    str
        Concatenated textual content.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for entry in content:
        text = getattr(entry, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _parse_arguments(arguments: Any) -> Any:
    """Parse tool/function call arguments when possible.

    Parameters
    ----------
    arguments : Any
        Argument payload supplied by the caller.

    Returns
    -------
    Any
        Parsed JSON object when the payload is a JSON string, otherwise the input value.
    """

    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _serialize_tool_calls(
    tool_calls: list[dict[str, Any]] | list[ChatCompletionMessageToolCall],
) -> list[dict[str, Any]]:
    """Convert tool calls into an Ollama JSON structure.

    Parameters
    ----------
    tool_calls : list[dict[str, Any]] | list[ChatCompletionMessageToolCall]
        Tool call payloads originating from OpenAI schemas or raw dicts.

    Returns
    -------
    list[dict[str, Any]]
        Normalized tool call representations compatible with Ollama responses.
    """

    serialized: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, ChatCompletionMessageToolCall):
            payload = {
                "type": call.type,
                "function": {
                    "name": call.function.name if call.function else None,
                    "arguments": _parse_arguments(
                        call.function.arguments if call.function else None
                    ),
                },
            }
        else:
            function = call.get("function", {})
            payload = {
                "type": call.get("type", "function"),
                "function": {
                    "name": function.get("name"),
                    "arguments": _parse_arguments(function.get("arguments")),
                },
            }
        serialized.append(payload)
    return serialized


def _usage_counts(usage: UsageInfo | dict[str, Any] | None) -> tuple[int, int]:
    """Extract prompt/completion token counts from usage metadata.

    Parameters
    ----------
    usage : UsageInfo | dict[str, Any] | None
        Usage metadata from a handler response.

    Returns
    -------
    tuple[int, int]
        Prompt token count and completion token count respectively.
    """

    prompt_tokens = 0
    completion_tokens = 0
    if usage is None:
        return prompt_tokens, completion_tokens
    if isinstance(usage, UsageInfo):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens or 0
        return prompt_tokens, completion_tokens
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return prompt_tokens, completion_tokens


def _build_common_stats(
    total_duration_ns: int, usage: UsageInfo | dict[str, Any] | None
) -> dict[str, Any]:
    """Build the shared stats payload for Ollama responses.

    Parameters
    ----------
    total_duration_ns : int
        Total elapsed duration in nanoseconds.
    usage : UsageInfo | dict[str, Any] | None
        Usage metadata including token counts.

    Returns
    -------
    dict[str, Any]
        Dictionary with duration and token statistics.
    """

    prompt_tokens, completion_tokens = _usage_counts(usage)
    return {
        "total_duration": total_duration_ns,
        "load_duration": 0,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": completion_tokens,
        "eval_duration": 0,
    }


def build_ollama_completion_payload(
    completion: ChatCompletionResponse,
    *,
    total_duration_ns: int,
    response_kind: ResponseKind,
) -> dict[str, Any]:
    """Convert a ChatCompletionResponse into an Ollama payload.

    Parameters
    ----------
    completion : ChatCompletionResponse
        Response returned by the handler.
    total_duration_ns : int
        Total completion duration.
    response_kind : ResponseKind
        Indicates whether the request was ``"chat"`` or ``"generate"``.

    Returns
    -------
    dict[str, Any]
        Serialized payload compatible with the Ollama REST schema.
    """

    choice = completion.choices[0]
    message = choice.message
    usage = completion.usage
    payload: dict[str, Any] = {
        "model": completion.model,
        "created_at": _iso_timestamp(),
        "done": True,
        "done_reason": choice.finish_reason or "stop",
        "context": [],
    }
    payload.update(_build_common_stats(total_duration_ns, usage))

    if response_kind == "chat":
        message_payload: dict[str, Any] = {
            "role": message.role,
            "content": _normalize_message_content(message.content),
            "images": None,
        }
        if message.reasoning_content:
            message_payload["thinking"] = message.reasoning_content
        tool_calls = _serialize_tool_calls(message.tool_calls or []) if message.tool_calls else None
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
        payload["message"] = message_payload
    else:
        payload["response"] = _normalize_message_content(message.content)
    return payload


def build_stream_final_payload(
    *,
    model: str,
    accumulated_content: str,
    total_duration_ns: int,
    usage: UsageInfo | dict[str, Any] | None,
    response_kind: ResponseKind,
    done_reason: str,
    tool_calls: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Create the terminal chunk for streaming responses.

    Parameters
    ----------
    model : str
        Model identifier used for the request.
    accumulated_content : str
        All content pieces concatenated across the stream.
    total_duration_ns : int
        Total elapsed duration in nanoseconds.
    usage : UsageInfo | dict[str, Any] | None
        Usage metadata captured during generation.
    response_kind : ResponseKind
        Distinguishes chat vs generate semantics.
    done_reason : str
        Finish reason reported by the handler.
    tool_calls : list[dict[str, Any]] | None
        Aggregated tool call payloads, if any.

    Returns
    -------
    dict[str, Any]
        Final chunk body for NDJSON streaming.
    """

    payload: dict[str, Any] = {
        "model": model,
        "created_at": _iso_timestamp(),
        "done": True,
        "done_reason": done_reason,
        "context": [],
    }
    payload.update(_build_common_stats(total_duration_ns, usage))
    if response_kind == "chat":
        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": accumulated_content,
            "images": None,
        }
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
        payload["message"] = message_payload
    else:
        payload["response"] = accumulated_content
    return payload


def _apply_ollama_options(
    chat_request: ChatCompletionRequest, options: dict[str, Any] | None
) -> ChatCompletionRequest:
    """Apply Ollama "options" overrides onto a chat request.

    Parameters
    ----------
    chat_request : ChatCompletionRequest
        Base request generated from incoming payloads.
    options : dict[str, Any] | None
        Optional Ollama "options" map to translate.

    Returns
    -------
    ChatCompletionRequest
        Updated request containing any supported option overrides.
    """

    if not isinstance(options, dict):
        return chat_request
    update: dict[str, Any] = {}
    for option_name, field_name in OPTION_FIELD_MAP.items():
        if option_name in options and options[option_name] is not None:
            update[field_name] = options[option_name]
    if not update:
        return chat_request
    return chat_request.model_copy(update=update)


def _prepare_response_format(format_spec: Any) -> dict[str, Any] | None:
    """Translate an Ollama response format spec into OpenAI schema.

    Parameters
    ----------
    format_spec : Any
        Request-level format entry provided by clients.

    Returns
    -------
    dict[str, Any] | None
        Response format dictionary suitable for ChatCompletionRequest or ``None``.
    """

    if isinstance(format_spec, str):
        if format_spec.lower() == "json":
            return {"type": "json_object"}
        return None
    if isinstance(format_spec, dict):
        return {"type": "json_schema", "json_schema": {"schema": format_spec}}
    return None


def _apply_request_extras(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Apply Ollama-specific extras onto a ChatCompletionRequest.

    Parameters
    ----------
    request : ChatCompletionRequest
        Request generated from incoming payloads.

    Returns
    -------
    ChatCompletionRequest
        Updated request with response format and option overrides applied.
    """

    extras = getattr(request, "model_extra", None)
    if not extras:
        return request
    updated_request = request
    response_format = _prepare_response_format(extras.get("format"))
    if response_format is not None:
        updated_request = updated_request.model_copy(update={"response_format": response_format})
    return _apply_ollama_options(updated_request, extras.get("options"))


def _coerce_keep_alive(value: Any) -> int | None:
    """Normalize keep_alive values into duration seconds or sentinels.

    Parameters
    ----------
    value : Any
        Raw keep-alive specification (string, numeric, etc.).

    Returns
    -------
    int | None
        ``0`` for immediate unload, positive seconds to hold the model,
        ``-1`` for indefinite hold, or ``None`` when no change is requested.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds < 0:
            return -1 if seconds == -1 else 0
        return seconds
    if isinstance(value, str):
        trimmed = value.strip().lower()
        if not trimmed:
            return None
        if trimmed in {"inf", "infinite", "forever", "-1"}:
            return -1
        if trimmed in {"true", "false"}:
            return None
        if trimmed in {"0", "0s", "0m", "0h"}:
            return 0
        unit = trimmed[-1]
        if unit in {"s", "m", "h"}:
            try:
                value_part = float(trimmed[:-1])
            except ValueError:
                return None
            multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
            seconds = int(value_part * multiplier)
            return max(seconds, 0)
        try:
            seconds = int(trimmed)
        except ValueError:
            return None
        if seconds < 0:
            return -1 if seconds == -1 else 0
        return seconds
    return None


async def _maybe_handle_keep_alive(raw_request: Request, keep_alive: Any, reason: str) -> None:
    """Handle keep_alive semantics for explicit load/unload timing.

    Parameters
    ----------
    raw_request : Request
        Incoming FastAPI request wrapper.
    keep_alive : Any
        Caller-supplied keep-alive value.
    reason : str
        Reason string recorded when unloading the handler.
    """

    normalized = _coerce_keep_alive(keep_alive)
    if normalized is None:
        return

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    controller = getattr(raw_request.app.state, "auto_unload_controller", None)
    if normalized == 0:
        if handler_manager is None:
            return
        await handler_manager.unload(reason)
        if controller is not None:
            controller.clear_hold()
        return

    if controller is None:
        return
    controller.request_hold(normalized)


def _chunk_to_content(chunk: Any) -> tuple[str, str]:
    """Split streaming chunks into content and reasoning pieces.

    Parameters
    ----------
    chunk : Any
        Streaming delta emitted by handlers.

    Returns
    -------
    tuple[str, str]
        Tuple of primary content and reasoning content.
    """

    if isinstance(chunk, str):
        return chunk, ""
    if isinstance(chunk, dict):
        content = chunk.get("content")
        reasoning_content = chunk.get("reasoning_content")
        if isinstance(content, str):
            return content, reasoning_content if isinstance(reasoning_content, str) else ""
        if isinstance(reasoning_content, str):
            return "", reasoning_content
    return "", ""


async def _stream_ollama_response(
    generator: AsyncGenerator[Any, None],
    *,
    model: str,
    response_kind: ResponseKind,
    raw_request: Request | None = None,
    keep_alive: Any = None,
    keep_alive_reason: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Convert handler streaming chunks into NDJSON responses.

    Parameters
    ----------
    generator : AsyncGenerator[Any, None]
        Handler-provided async generator yielding chunks.
    model : str
        Active model identifier.
    response_kind : ResponseKind
        Indicates chat vs generate streaming semantics.
    raw_request : Request | None, optional
        Original FastAPI request for keep-alive handling.
    keep_alive : Any, optional
        Keep-alive metadata provided by the client.
    keep_alive_reason : str | None, optional
        Reason message recorded when unloading after streaming.

    Yields
    ------
    AsyncGenerator[bytes, None]
        NDJSON-encoded streaming payloads.
    """
    aggregated_content: list[str] = []
    usage_summary: dict[str, Any] | None = None
    tool_calls: dict[int, dict[str, Any]] = {}
    done_reason = "stop"
    start_ns = time.perf_counter_ns()
    had_error = False

    try:
        async for chunk in generator:
            if chunk is None:
                continue
            if isinstance(chunk, dict) and "__usage__" in chunk:
                usage_summary = chunk["__usage__"]
                continue

            if isinstance(chunk, dict) and ("name" in chunk or "arguments" in chunk):
                index = int(chunk.get("index", len(tool_calls)))
                entry = tool_calls.setdefault(
                    index,
                    {"type": "function", "function": {"name": None, "arguments": ""}},
                )
                function_payload = entry["function"]
                if "name" in chunk:
                    function_payload["name"] = chunk["name"]
                    done_reason = "tool_calls"
                if "arguments" in chunk:
                    existing_args = function_payload.get("arguments", "") or ""
                    function_payload["arguments"] = f"{existing_args}{chunk['arguments']}"
                payload: dict[str, Any] = {
                    "model": model,
                    "created_at": _iso_timestamp(),
                    "done": False,
                }
                if response_kind == "chat":
                    payload["message"] = {
                        "role": "assistant",
                        "content": "",
                        "images": None,
                        "tool_calls": _serialize_tool_calls(list(tool_calls.values())),
                    }
                yield json.dumps(payload).encode("utf-8") + b"\n"
                continue

            content_piece, reasoning = _chunk_to_content(chunk)
            if not content_piece and not reasoning:
                continue
            if content_piece:
                aggregated_content.append(content_piece)

            payload = {
                "model": model,
                "created_at": _iso_timestamp(),
                "done": False,
            }
            if response_kind == "chat":
                message_payload: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_piece,
                    "images": None,
                }
                if reasoning:
                    message_payload["thinking"] = reasoning
                if tool_calls:
                    message_payload["tool_calls"] = _serialize_tool_calls(list(tool_calls.values()))
                payload["message"] = message_payload
            else:
                payload["response"] = content_piece
            yield json.dumps(payload).encode("utf-8") + b"\n"
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        had_error = True
        logger.exception(f"Error streaming Ollama response: {type(exc).__name__}: {exc}")
        error_payload = create_error_response(
            str(exc), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR
        )
        yield (json.dumps(error_payload) + "\n").encode("utf-8")
    finally:
        total_duration_ns = int(time.perf_counter_ns() - start_ns)
        if not had_error:
            final_payload = build_stream_final_payload(
                model=model,
                accumulated_content="".join(aggregated_content),
                total_duration_ns=total_duration_ns,
                usage=usage_summary,
                response_kind=response_kind,
                done_reason=done_reason,
                tool_calls=_serialize_tool_calls(list(tool_calls.values())) if tool_calls else None,
            )
            yield json.dumps(final_payload).encode("utf-8") + b"\n"
        if raw_request is not None and keep_alive_reason is not None:
            await _maybe_handle_keep_alive(raw_request, keep_alive, keep_alive_reason)


async def _run_completion(
    handler: MLXHandlerType,
    request: ChatCompletionRequest,
    raw_request: Request,
    *,
    response_kind: ResponseKind,
    keep_alive: Any = None,
    keep_alive_reason: str | None = None,
) -> JSONResponse | StreamingResponse:
    """Execute a completion against the provided handler.

    Parameters
    ----------
    handler : MLXHandlerType
        Loaded handler instance used to service the request.
    request : ChatCompletionRequest
        Normalized chat/generate request.
    raw_request : Request
        FastAPI request, used for metadata lookups.
    response_kind : ResponseKind
        Indicates chat vs generate response shaping.
    keep_alive : Any, optional
        Optional keep-alive flag passed through for streaming responses.
    keep_alive_reason : str | None, optional
        Reason string logged when unloading after streaming.

    Returns
    -------
    JSONResponse | StreamingResponse
        FastAPI response ready to return to the caller.
    """
    start_ns = time.perf_counter_ns()
    request_id = getattr(raw_request.state, "request_id", None)

    if request.stream:
        if isinstance(handler, MLXVLMHandler):
            generator = handler.generate_multimodal_stream(request)
        else:
            generator = handler.generate_text_stream(request)  # type: ignore[union-attr]
        return StreamingResponse(
            _stream_ollama_response(
                generator,
                model=request.model,
                response_kind=response_kind,
                raw_request=raw_request,
                keep_alive=keep_alive,
                keep_alive_reason=keep_alive_reason,
            ),
            media_type=NDJSON_MEDIA_TYPE,
            headers=NDJSON_HEADERS,
        )

    if isinstance(handler, MLXVLMHandler):
        result = await handler.generate_multimodal_response(request)
        if isinstance(result, dict) and "response" in result and "usage" in result:
            completion = format_final_response(
                result["response"], request.model, request_id, result.get("usage")
            )
        else:
            completion = format_final_response(result, request.model, request_id)
    else:
        result = await handler.generate_text_response(request)  # type: ignore[union-attr]
        text_response = result.get("response")
        if text_response is None:
            msg = "Text handler response missing 'response' payload"
            raise ValueError(msg)
        completion = format_final_response(
            text_response,
            request.model,
            request_id,
            result.get("usage"),
        )

    total_duration = int(time.perf_counter_ns() - start_ns)
    payload = build_ollama_completion_payload(
        completion,
        total_duration_ns=total_duration,
        response_kind=response_kind,
    )
    return JSONResponse(payload)


def _convert_metadata_to_tag(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert internal model metadata into an Ollama tag payload.

    Parameters
    ----------
    entry : dict[str, Any]
        Metadata entry stored on ``app.state``.

    Returns
    -------
    dict[str, Any]
        Tag dictionary compatible with ``/api/tags``.
    """

    metadata = entry.get("metadata", {})
    model_id = entry.get("id") or metadata.get("model_path") or "local-model"
    model_type = metadata.get("model_type", "mlx")
    families = [model_type] if model_type else []
    details = {
        "parent_model": metadata.get("parent_model", ""),
        "format": metadata.get("format", "mlx"),
        "family": model_type,
        "families": families,
        "parameter_size": metadata.get("parameter_size"),
        "quantization_level": metadata.get("quantization_level"),
    }
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": _iso_timestamp(entry.get("created")),
        "size": metadata.get("size"),
        "digest": metadata.get("digest"),
        "details": details,
    }


def _current_model_descriptor(raw_request: Request) -> dict[str, Any]:
    """Build a descriptor of the current configured model.

    Parameters
    ----------
    raw_request : Request
        FastAPI request containing application state.

    Returns
    -------
    dict[str, Any]
        Descriptor describing the configured model.
    """

    config = getattr(raw_request.app.state, "server_config", None)
    identifier = getattr(config, "model_identifier", None) or getattr(
        config, "model_path", "local-model"
    )
    model_type = getattr(config, "model_type", "mlx")
    return {
        "name": identifier,
        "model": identifier,
        "modified_at": _iso_timestamp(),
        "size": None,
        "digest": None,
        "details": {
            "parent_model": "",
            "format": "mlx",
            "family": model_type,
            "families": [model_type],
            "parameter_size": None,
            "quantization_level": getattr(config, "quantize", None),
        },
    }


def _collect_model_tags(raw_request: Request) -> list[dict[str, Any]]:
    """Collect the tag list reported by ``/api/tags``.

    Parameters
    ----------
    raw_request : Request
        FastAPI request storing cached metadata.

    Returns
    -------
    list[dict[str, Any]]
        List of model descriptors exposed to clients.
    """

    metadata = getattr(raw_request.app.state, "model_metadata", [])
    if metadata:
        return [_convert_metadata_to_tag(metadata[0])]
    return [_current_model_descriptor(raw_request)]


def _collect_running_models(raw_request: Request) -> list[dict[str, Any]]:
    """Report currently loaded models for ``/api/ps``.

    Parameters
    ----------
    raw_request : Request
        FastAPI request storing handler manager state.

    Returns
    -------
    list[dict[str, Any]]
        Model descriptors of loaded handlers, if any.
    """

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    if handler_manager and handler_manager.current_handler:
        descriptor = _current_model_descriptor(raw_request)
        descriptor["expires_at"] = None
        descriptor["size_vram"] = None
        return [descriptor]
    return []


def _embedding_input_count(embedding_input: list[str] | str) -> int:
    """Compute how many embedding inputs were provided.

    Parameters
    ----------
    embedding_input : list[str] | str
        Original embedding input payload.

    Returns
    -------
    int
        Number of embedding inputs.
    """

    return len(embedding_input) if isinstance(embedding_input, list) else 1


async def _produce_embeddings(
    handler: MLXEmbeddingsHandler, request: EmbeddingRequest
) -> tuple[list[list[float]], int]:
    """Generate embeddings and capture total duration.

    Parameters
    ----------
    handler : MLXEmbeddingsHandler
        Handler responsible for embeddings generation.
    request : EmbeddingRequest
        Normalized embedding request.

    Returns
    -------
    tuple[list[list[float]], int]
        Tuple of embeddings list and elapsed duration in nanoseconds.
    """

    start_ns = time.perf_counter_ns()
    embeddings = await handler.generate_embeddings_response(request)
    return embeddings, int(time.perf_counter_ns() - start_ns)


@router.get("/tags", response_model=None)
async def list_models(raw_request: Request) -> dict[str, Any]:
    """List available models in Ollama ``/api/tags`` shape.

    Parameters
    ----------
    raw_request : Request
        FastAPI request providing application state access.

    Returns
    -------
    dict[str, Any]
        Payload with ``models`` describing each available tag.
    """

    return {"models": _collect_model_tags(raw_request)}


@router.get("/ps", response_model=None)
async def list_running_models(raw_request: Request) -> dict[str, Any]:
    """List currently loaded models (best effort).

    Parameters
    ----------
    raw_request : Request
        FastAPI request providing handler manager access.

    Returns
    -------
    dict[str, Any]
        Payload containing ``models`` currently loaded.
    """

    return {"models": _collect_running_models(raw_request)}


@router.get("/version", response_model=None)
async def version() -> dict[str, str]:
    """Expose the server version via ``/api/version``.

    Returns
    -------
    dict[str, str]
        Dictionary containing the semantic version string.
    """

    return {"version": __version__}


@router.post("/chat", response_model=None)
async def chat(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> JSONResponse | StreamingResponse:
    """Handle ``/api/chat`` requests using MLX handlers.

    Parameters
    ----------
    request : ChatCompletionRequest
        Chat request payload.
    raw_request : Request
        FastAPI request exposing state references.

    Returns
    -------
    JSONResponse | StreamingResponse
        Response mirroring Ollama's chat endpoint semantics.
    """

    handler, error = await _get_handler_or_error(raw_request, "ollama_chat")
    if error is not None:
        return error
    if handler is None or not isinstance(handler, (MLXLMHandler, MLXVLMHandler)):
        return JSONResponse(
            content=create_error_response(
                "Chat requests require a language or multimodal model.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    request = _apply_request_extras(request)
    extras = getattr(request, "model_extra", None) or {}
    keep_alive_value = extras.get("keep_alive")
    response = await _run_completion(
        handler,
        request,
        raw_request,
        response_kind="chat",
        keep_alive=keep_alive_value,
        keep_alive_reason="ollama_chat_keep_alive",
    )
    if not request.stream:
        await _maybe_handle_keep_alive(raw_request, keep_alive_value, "ollama_chat_keep_alive")
    return response


@router.post("/generate", response_model=None)
async def generate(
    request: OllamaGenerateRequest,
    raw_request: Request,
) -> JSONResponse | StreamingResponse:
    """Handle ``/api/generate`` requests by constructing chat payloads.

    Parameters
    ----------
    request : OllamaGenerateRequest
        Generate payload supplied by the client.
    raw_request : Request
        FastAPI request for accessing handler state.

    Returns
    -------
    JSONResponse | StreamingResponse
        Response object compatible with Ollama's generate endpoint.
    """

    chat_request = request.to_chat_request()
    handler, error = await _get_handler_or_error(raw_request, "ollama_generate")
    if error is not None:
        return error
    if handler is None or not isinstance(handler, (MLXLMHandler, MLXVLMHandler)):
        return JSONResponse(
            content=create_error_response(
                "Generate requests require a language or multimodal model.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    response = await _run_completion(
        handler,
        chat_request,
        raw_request,
        response_kind="generate",
        keep_alive=request.keep_alive,
        keep_alive_reason="ollama_generate_keep_alive",
    )
    if not chat_request.stream:
        await _maybe_handle_keep_alive(
            raw_request, request.keep_alive, "ollama_generate_keep_alive"
        )
    return response


@router.post("/embed", response_model=None)
async def embed(
    request: EmbeddingRequest,
    raw_request: Request,
) -> JSONResponse | HTTPException:
    """Handle ``/api/embed`` embedding requests.

    Parameters
    ----------
    request : EmbeddingRequest
        Embedding request body.
    raw_request : Request
        FastAPI request giving access to the handler manager.

    Returns
    -------
    JSONResponse | HTTPException
        Embedding response payload or error.
    """

    handler, error = await _get_handler_or_error(raw_request, "ollama_embed")
    if error is not None:
        return error
    if handler is None or not isinstance(handler, MLXEmbeddingsHandler):
        return JSONResponse(
            content=create_error_response(
                "Embedding requests require --model-type embeddings.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    embeddings, duration = await _produce_embeddings(handler, request)
    payload = {
        "model": request.model,
        "embeddings": embeddings,
        "total_duration": duration,
        "load_duration": 0,
        "prompt_eval_count": _embedding_input_count(request.input),
    }
    return JSONResponse(payload)


@router.post("/embeddings", response_model=None)
async def embeddings_legacy(
    request: EmbeddingRequest,
    raw_request: Request,
) -> JSONResponse | HTTPException:
    """Compatibility endpoint for the legacy ``/api/embeddings`` route.

    Parameters
    ----------
    request : EmbeddingRequest
        Embedding request body.
    raw_request : Request
        FastAPI request providing handler manager access.

    Returns
    -------
    JSONResponse | HTTPException
        Response shaped like the legacy embeddings endpoint.
    """

    handler, error = await _get_handler_or_error(raw_request, "ollama_embeddings")
    if error is not None:
        return error
    if handler is None or not isinstance(handler, MLXEmbeddingsHandler):
        return JSONResponse(
            content=create_error_response(
                "Embedding requests require --model-type embeddings.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    embeddings, duration = await _produce_embeddings(handler, request)
    vector = embeddings[0] if embeddings else []
    payload = {
        "model": request.model,
        "embedding": vector,
        "total_duration": duration,
        "load_duration": 0,
        "prompt_eval_count": _embedding_input_count(request.input),
    }
    return JSONResponse(payload)


@router.post("/load", response_model=None)
async def load_model(raw_request: Request, request: LifecycleRequest) -> JSONResponse:
    """Explicitly load the configured model into memory.

    Parameters
    ----------
    raw_request : Request
        FastAPI request exposing the handler manager.
    request : LifecycleRequest
        Body containing optional model identifiers.

    Returns
    -------
    JSONResponse
        Confirmation payload describing the load event.
    """

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    if handler_manager is None:
        return JSONResponse(
            content=create_error_response(
                "Handler manager not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    await handler_manager.ensure_loaded("ollama_load_endpoint")
    model_name = request.model or handler_manager.config_args.model_identifier
    payload = {
        "model": model_name,
        "created_at": _iso_timestamp(),
        "message": {"role": "assistant", "content": ""},
        "done_reason": "load",
        "done": True,
    }
    return JSONResponse(payload)


@router.post("/unload", response_model=None)
async def unload_model(raw_request: Request, request: LifecycleRequest) -> JSONResponse:
    """Explicitly unload the configured model from memory.

    Parameters
    ----------
    raw_request : Request
        FastAPI request exposing the handler manager.
    request : LifecycleRequest
        Body containing optional model identifiers.

    Returns
    -------
    JSONResponse
        Confirmation payload describing the unload event.
    """

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    if handler_manager is None:
        return JSONResponse(
            content=create_error_response(
                "Handler manager not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
    await handler_manager.unload("ollama_unload_endpoint")
    model_name = request.model or handler_manager.config_args.model_identifier
    payload = {
        "model": model_name,
        "created_at": _iso_timestamp(),
        "message": {"role": "assistant", "content": ""},
        "done_reason": "unload",
        "done": True,
    }
    return JSONResponse(payload)
