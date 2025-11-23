"""Tests for hub service-backed FastAPI endpoints."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.hub_routes import hub_router
from app.hub.service import HubServiceError
from app.server import configure_fastapi_app


class _StubServiceState:
    def __init__(self) -> None:
        self.available = True
        self.reload_calls = 0
        self.reload_result = {"started": [], "stopped": [], "unchanged": []}
        self.status_payload: dict[str, Any] = {
            "models": [],
            "timestamp": 1,
        }
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.shutdown_called = False


class _StubServiceClient:
    state: _StubServiceState

    def __init__(self, _socket_path: str, *, timeout: float = 5.0) -> None:  # noqa: ARG002
        if not hasattr(_StubServiceClient, "state"):
            raise RuntimeError("Stub state not configured")

    def is_available(self) -> bool:
        return _StubServiceClient.state.available

    def wait_until_available(self, timeout: float = 10.0) -> bool:  # noqa: ARG002
        return self.is_available()

    def reload(self) -> dict[str, Any]:
        if not self.is_available():
            raise HubServiceError("unavailable")
        _StubServiceClient.state.reload_calls += 1
        return _StubServiceClient.state.reload_result

    def status(self) -> dict[str, Any]:
        if not self.is_available():
            raise HubServiceError("unavailable")
        return _StubServiceClient.state.status_payload

    def start_model(self, name: str) -> None:
        if not self.is_available():
            raise HubServiceError("unavailable")
        if name == "saturated":
            raise HubServiceError("group full", status_code=HTTPStatus.TOO_MANY_REQUESTS)
        _StubServiceClient.state.start_calls.append(name)

    def stop_model(self, name: str) -> None:
        if not self.is_available():
            raise HubServiceError("unavailable")
        _StubServiceClient.state.stop_calls.append(name)

    def shutdown(self) -> None:
        if not self.is_available():
            raise HubServiceError("unavailable")
        _StubServiceClient.state.shutdown_called = True


@pytest.fixture
def hub_service_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, _StubServiceState]:
    """Return a TestClient configured with a stubbed hub service backend."""
    config_dir = tmp_path / "hub-config"
    config_dir.mkdir()
    config_path = config_dir / "hub.yaml"
    config_path.write_text(
        """
host: 0.0.0.0
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
    model_type: lm
    default: true
  - name: beta
    model_path: /models/beta
    model_type: lm
    group: test
""".strip()
    )

    app = FastAPI()
    configure_fastapi_app(app)
    app.include_router(hub_router)
    app.state.server_config = SimpleNamespace(enable_status_page=True, host="0.0.0.0", port=8123)
    app.state.hub_config_path = config_path

    state = _StubServiceState()
    state.status_payload["models"] = [
        {
            "name": "alpha",
            "state": "running",
            "pid": 111,
            "group": "default",
            "log_path": "/tmp/alpha.log",
        }
    ]

    _StubServiceClient.state = state
    monkeypatch.setattr("app.api.hub_routes.HubServiceClient", _StubServiceClient)

    def _fake_start_process(path: str) -> int:  # noqa: ARG001
        state.available = True
        return 4321

    monkeypatch.setattr("app.api.hub_routes.start_hub_service_process", _fake_start_process)

    client = TestClient(app)
    try:
        yield client, state
    finally:
        client.close()


def test_hub_status_uses_service_snapshot(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """Hub status endpoint should prefer hub service snapshots when available."""
    client, state = hub_service_app
    state.available = True

    response = client.get("/hub/status")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["counts"] == {"registered": 2, "loaded": 1}
    assert payload["models"][0]["metadata"]["default"] is True
    assert payload["models"][0]["metadata"]["status"] == "running"
    assert state.reload_calls == 1


def test_hub_status_degrades_when_service_unavailable(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """Hub status should downgrade to 'degraded' when the service is offline."""
    client, state = hub_service_app
    state.available = False

    response = client.get("/hub/status")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["counts"]["registered"] == 2


def test_hub_model_load_calls_service_client(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """Model load endpoint should call HubServiceClient.start_model after reload."""
    client, state = hub_service_app
    state.available = True

    response = client.post("/hub/models/alpha/load", json={})

    assert response.status_code == HTTPStatus.OK
    assert state.start_calls == ["alpha"]
    assert state.reload_calls == 1


def test_hub_model_load_surfaces_capacity_errors(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """Capacity errors from HubServiceClient should translate to HTTP 429 responses."""
    client, state = hub_service_app
    state.available = True

    response = client.post("/hub/models/saturated/load", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    body = response.json()
    assert body["error"]["message"].startswith("Failed to load")


def test_hub_service_start_spawns_process_when_missing(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """/hub/service/start should spawn the service when it is not running."""
    client, state = hub_service_app
    state.available = False

    response = client.post("/hub/service/start")

    assert response.status_code == HTTPStatus.OK
    assert state.reload_calls == 1  # reload invoked after start
    body = response.json()
    assert body["action"] == "start"
    assert body["details"]["pid"] == 4321


def test_hub_service_stop_requires_running_manager(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """Stopping the service should fail cleanly when it is not running."""

    client, state = hub_service_app
    state.available = False

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert state.shutdown_called is False


def test_hub_service_reload_endpoint_returns_diff(
    hub_service_app: tuple[TestClient, _StubServiceState],
) -> None:
    """/hub/service/reload should surface the diff returned by the service."""

    client, state = hub_service_app
    state.available = True
    state.reload_result = {"started": ["alpha"], "stopped": ["beta"], "unchanged": []}

    response = client.post("/hub/service/reload")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["action"] == "reload"
    assert body["details"] == state.reload_result
    assert state.reload_calls == 1
