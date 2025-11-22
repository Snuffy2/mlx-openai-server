"""Tests for router registration based on the configured API mode."""

from __future__ import annotations

from app.config import MLXServerConfig
from app.server import setup_server


def _collect_paths(config: MLXServerConfig) -> set[str]:
    server_config = setup_server(config)
    assert hasattr(server_config, "app"), "setup_server should return a uvicorn.Config with app"
    app = server_config.app  # type: ignore[assignment]
    return {route.path for route in app.routes}


def test_setup_server_registers_only_openai_when_requested() -> None:
    """When api_mode=openai only OpenAI routes should be mounted."""

    config = MLXServerConfig(model_path="dummy", api_mode="openai")
    paths = _collect_paths(config)
    assert "/v1/chat/completions" in paths
    assert "/api/chat" not in paths


def test_setup_server_registers_only_ollama_when_requested() -> None:
    """When api_mode=ollama only Ollama routes should be mounted."""

    config = MLXServerConfig(model_path="dummy", api_mode="ollama")
    paths = _collect_paths(config)
    assert "/api/chat" in paths
    assert "/v1/chat/completions" not in paths


def test_setup_server_registers_both_when_requested() -> None:
    """When api_mode=both both router sets should be mounted."""

    config = MLXServerConfig(model_path="dummy", api_mode="both")
    paths = _collect_paths(config)
    assert "/v1/chat/completions" in paths
    assert "/api/chat" in paths
