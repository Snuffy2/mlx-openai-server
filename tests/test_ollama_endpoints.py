"""Tests for lightweight Ollama endpoint helpers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.api import ollama_endpoints as ollama


def _build_request(**state_kwargs: object) -> SimpleNamespace:
    state = SimpleNamespace(**state_kwargs)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_list_models_uses_metadata_entry() -> None:
    """The /api/tags helper should prefer existing metadata entries."""

    metadata_entry = {
        "id": "test-model",
        "created": 0,
        "metadata": {"model_type": "lm", "model_path": "test-model"},
    }
    request = _build_request(model_metadata=[metadata_entry])

    response = asyncio.run(ollama.list_models(request))

    assert response["models"][0]["name"] == "test-model"


def test_list_running_models_reports_loaded_handler() -> None:
    """The /api/ps helper should report running models when present."""

    server_config = SimpleNamespace(model_identifier="demo", model_type="lm", quantize=4)
    handler_manager = SimpleNamespace(current_handler=object())
    request = _build_request(server_config=server_config, handler_manager=handler_manager)

    response = asyncio.run(ollama.list_running_models(request))

    assert response["models"][0]["name"] == "demo"


class _DummyHandlerManager:
    def __init__(self) -> None:
        self.config_args = SimpleNamespace(model_identifier="configured-model")
        self.ensure_loaded_calls: list[str] = []
        self.unload_calls: list[str] = []

    async def ensure_loaded(self, reason: str) -> None:
        self.ensure_loaded_calls.append(reason)

    async def unload(self, reason: str) -> None:
        self.unload_calls.append(reason)


class _DummyAutoUnloadController:
    def __init__(self) -> None:
        self.requested: list[int] = []
        self.clear_count = 0

    def request_hold(self, seconds: int) -> None:
        self.requested.append(seconds)

    def clear_hold(self) -> None:
        self.clear_count += 1


def test_load_model_invokes_handler_manager() -> None:
    """/api/load should trigger handler_manager.ensure_loaded and echo metadata."""

    handler_manager = _DummyHandlerManager()
    request = _build_request(handler_manager=handler_manager)
    lifecycle_request = ollama.LifecycleRequest(model=None)

    response = asyncio.run(ollama.load_model(request, lifecycle_request))

    assert handler_manager.ensure_loaded_calls == ["ollama_load_endpoint"]
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["done_reason"] == "load"
    assert payload["model"] == "configured-model"


def test_unload_model_invokes_handler_manager() -> None:
    """/api/unload should trigger handler_manager.unload and echo metadata."""

    handler_manager = _DummyHandlerManager()
    request = _build_request(handler_manager=handler_manager)
    lifecycle_request = ollama.LifecycleRequest(model="custom")

    response = asyncio.run(ollama.unload_model(request, lifecycle_request))

    assert handler_manager.unload_calls == ["ollama_unload_endpoint"]
    payload = json.loads(response.body.decode("utf-8"))
    assert payload["done_reason"] == "unload"
    assert payload["model"] == "custom"


def test_coerce_keep_alive_parses_duration_and_indefinite() -> None:
    """_coerce_keep_alive should parse strings similar to Ollama semantics."""

    assert ollama._coerce_keep_alive("30s") == 30
    assert ollama._coerce_keep_alive("2m") == 120
    assert ollama._coerce_keep_alive("1h") == 3600
    assert ollama._coerce_keep_alive("-1") == -1


def test_keep_alive_duration_requests_hold() -> None:
    """Positive keep_alive values should request hold on the controller."""

    controller = _DummyAutoUnloadController()
    request = _build_request(auto_unload_controller=controller, handler_manager=None)

    asyncio.run(ollama._maybe_handle_keep_alive(request, "45s", "reason"))

    assert controller.requested == [45]


def test_keep_alive_zero_unloads_and_clears_hold() -> None:
    """keep_alive=0 should unload immediately and clear controller hold."""

    handler_manager = _DummyHandlerManager()
    controller = _DummyAutoUnloadController()
    request = _build_request(
        handler_manager=handler_manager,
        auto_unload_controller=controller,
    )

    asyncio.run(ollama._maybe_handle_keep_alive(request, 0, "reason"))

    assert handler_manager.unload_calls == ["reason"]
    assert controller.clear_count == 1
