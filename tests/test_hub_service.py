"""Tests for hub service-backed FastAPI endpoints."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from app.api.hub_routes import HubServiceError, hub_router
from app.core.model_registry import ModelRegistry
from app.hub.daemon import HubSupervisor
from app.server import _hub_sync_once, configure_fastapi_app

# Stub classes `_StubServiceState` and `_StubController` are provided by `tests.conftest`

if TYPE_CHECKING:
    # Import test-only types for annotations without causing runtime imports
    from tests.conftest import _StubController, _StubServiceState


@pytest.fixture
def hub_service_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_service_state: _StubServiceState,
    stub_controller: _StubController,
) -> tuple[TestClient, Any, Any]:
    """
    Provide a TestClient configured with a stubbed hub service backend for testing hub endpoints.

    Sets up a FastAPI app with hub routes, writes a temporary hub.yaml configuration, attaches the provided stub controller and stub service state to app.state, and monkeypatches daemon interactions and process start/stop helpers so tests can simulate service availability, reloads, model start/stop, and shutdown. Yields a tuple of the test client, the stub service state, and the stub controller; ensures the client is closed on teardown.

    Parameters:
        tmp_path (Path): Temporary filesystem path for writing the hub configuration and logs.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch daemon and process functions.
        stub_service_state (_StubServiceState): Mutable stub representing daemon state, call counts, and simulated responses.
        stub_controller (_StubController): Stub controller used to observe controller interactions (reloads, start/stop, loads).

    Returns:
        tuple[TestClient, Any, Any]: A TestClient for the configured FastAPI app, the stub service state, and the stub controller.
    """
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
""".strip(),
    )

    app = FastAPI()
    configure_fastapi_app(app)
    app.include_router(hub_router)
    app.state.server_config = SimpleNamespace(enable_status_page=True, host="0.0.0.0", port=8123)
    app.state.hub_config_path = config_path
    controller = stub_controller
    app.state.hub_controller = controller

    state = stub_service_state
    state.status_payload["models"] = [
        {
            "name": "alpha",
            "state": "running",
            "pid": 111,
            "group": "default",
            "log_path": str(tmp_path / "alpha.log"),
        },
    ]

    async def _stub_call(
        config: Any,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        # Emulate the daemon HTTP surface used by the routes.
        """
        Emulate the daemon HTTP API used by the hub routes for testing.

        Simulates responses and side effects for a small set of endpoints:
        - GET /health: returns {"status": "ok"} when the stubbed service is available.
        - POST /hub/reload: increments state.reload_calls and returns state.reload_result when available.
        - GET /hub/status: returns state.status_payload when available.
        - POST /hub/models/{name}/start: records the start attempt in state.start_calls and returns {"message": "started"}; if name == "saturated" raises a capacity error with HTTP 429.
        - POST /hub/models/{name}/stop: records the stop attempt in state.stop_calls and returns {"message": "stopped"}.
        - POST /hub/shutdown: marks state.shutdown_called and returns {"message": "shutdown"}.
        For any other method/path combination this stub raises a HubServiceError.

        Parameters:
            config (Any): Ignored; present to match the real daemon call signature.
            method (str): HTTP method of the simulated request (e.g., "GET", "POST").
            path (str): Request path to simulate.
            json (Any | None): Ignored payload, accepted for signature compatibility.
            timeout (float): Ignored timeout, accepted for signature compatibility.

        Returns:
            dict[str, Any]: The simulated JSON response for the requested endpoint.

        Raises:
            HubServiceError: When the stubbed service is marked unavailable, when an unhandled path/method is requested, or to simulate a capacity error (status code 429) for starting the "saturated" model.
        """
        if method == "GET" and path == "/health":
            if not state.available:
                raise HubServiceError("unavailable")
            return {"status": "ok"}
        if method == "POST" and path == "/hub/reload":
            if not state.available:
                raise HubServiceError("unavailable")
            state.reload_calls += 1
            return state.reload_result
        if method == "GET" and path == "/hub/status":
            if not state.available:
                raise HubServiceError("unavailable")
            return state.status_payload
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            if not state.available:
                raise HubServiceError("unavailable")
            name = path.split("/")[-2]
            if name == "saturated":
                raise HubServiceError("group full", status_code=HTTPStatus.TOO_MANY_REQUESTS)
            state.start_calls.append(name)
            return {"message": "started"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            if not state.available:
                raise HubServiceError("unavailable")
            name = path.split("/")[-2]
            state.stop_calls.append(name)
            return {"message": "stopped"}
        if method == "POST" and path == "/hub/shutdown":
            if not state.available:
                raise HubServiceError("unavailable")
            state.shutdown_called = True
            return {"message": "shutdown"}
        raise HubServiceError("unhandled stub call")

    monkeypatch.setattr("app.api.hub_routes._call_daemon_api_async", _stub_call)

    def _fake_start_process(path: str, *, host: str | None = None, port: int | None = None) -> int:  # noqa: ARG001
        """
        Replaceable test helper that simulates starting the hub service process by marking the service available.

        Parameters:
            path (str): Path to the hub executable (accepted but not used by the stub).
            host (str | None): Host for the service (optional; accepted but ignored).
            port (int | None): Port for the service (optional; accepted but ignored).

        Returns:
            int: A fake process ID (4321).
        """
        state.available = True
        return 4321

    monkeypatch.setattr("app.api.hub_routes.start_hub_service_process", _fake_start_process)

    def _fake_stop_controller(_config: Any) -> bool:  # noqa: ANN401
        """
        Simulates stopping the controller and records that a stop was requested.

        Parameters:
            _config (Any): Ignored; present to match the real stop function signature.

        Returns:
            bool: `True` to indicate the simulated stop succeeded.
        """
        state.controller_stop_calls += 1
        return True

    monkeypatch.setattr("app.api.hub_routes._stop_controller_process", _fake_stop_controller)

    client = TestClient(app)
    try:
        yield client, state, controller
    finally:
        client.close()


def test_hub_status_uses_service_snapshot(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Hub status endpoint should prefer hub service snapshots when available."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.get("/hub/status")

    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["counts"] == {"registered": 2, "started": 1, "loaded": 1}
    assert payload["models"][0]["metadata"]["default"] is True
    assert payload["models"][0]["metadata"]["status"] == "running"
    assert controller.reload_count == 1


def test_hub_status_ok_when_service_unavailable_and_controller_available(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Hub status should be ok when controller is available, even if service is offline."""
    client, state, _controller = hub_service_app
    state.available = False

    response = client.get("/hub/status")
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["counts"]["registered"] == 2


def test_hub_model_start_calls_service_client(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Model start endpoint should call controller.start_model."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.post("/hub/models/alpha/start", json={})

    assert response.status_code == HTTPStatus.OK
    assert controller.started == ["alpha"]
    assert state.reload_calls == 0


def test_hub_model_start_surfaces_capacity_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Capacity errors from controller should translate to HTTP 429 responses."""
    client, state, _controller = hub_service_app
    state.available = True

    response = client.post("/hub/models/saturated/start", json={})

    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    body = response.json()
    assert body["error"]["message"] == "429: group full"


def test_hub_service_start_spawns_process_when_missing(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/service/start should spawn the service when it is not running."""
    client, state, _controller = hub_service_app
    state.available = False

    response = client.post("/hub/service/start")

    assert response.status_code == HTTPStatus.OK
    assert state.reload_calls == 1  # reload invoked after start
    body = response.json()
    assert body["action"] == "start"
    assert body["details"]["pid"] == 4321


def test_hub_service_stop_handles_missing_manager(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Stop should still return success when the manager is offline.

    Parameters
    ----------
    hub_service_app : tuple[TestClient, _StubServiceState, _StubController]
        Fixture providing the test client and stubs.
    """
    client, state, _controller = hub_service_app
    state.available = False

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.OK
    assert state.shutdown_called is False
    assert state.controller_stop_calls == 1
    payload = response.json()
    assert payload["details"] == {"controller_stopped": True, "manager_shutdown": False}


def test_hub_service_stop_shuts_down_manager_when_available(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Stop should mirror CLI behavior by halting controller and manager.

    Parameters
    ----------
    hub_service_app : tuple[TestClient, _StubServiceState, _StubController]
        Fixture providing the test client and stubs.
    """
    client, state, _controller = hub_service_app
    state.available = True

    response = client.post("/hub/service/stop")

    assert response.status_code == HTTPStatus.OK
    assert state.shutdown_called is True
    assert state.reload_calls == 1
    assert state.controller_stop_calls == 1
    payload = response.json()
    assert payload["details"] == {"controller_stopped": True, "manager_shutdown": True}


def test_hub_service_reload_endpoint_returns_diff(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/service/reload should surface the diff returned by the service."""
    client, state, controller = hub_service_app
    state.available = True

    response = client.post("/hub/service/reload")

    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["action"] == "reload"
    assert body["details"] == {
        "started": [],
        "stopped": ["old_model"],
        "unchanged": ["alpha", "beta"],
    }
    assert controller.reload_count == 1


def test_hub_load_model_invokes_controller(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """/hub/models/{model}/load should call the controller."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/alpha/load", json={"reason": "dashboard"})

    assert response.status_code == HTTPStatus.OK
    assert controller.loaded == ["alpha"]


def test_hub_memory_actions_surface_controller_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Controller-originated failures should propagate to the client."""
    client, _state, controller = hub_service_app

    response = client.post("/hub/models/denied/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS
    payload = response.json()
    assert "group busy" in payload["error"]["message"]

    response = client.post("/hub/models/missing/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert controller.unloaded[-1] == "missing"


def test_vram_admin_endpoints_invoke_registry(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Admin VRAM endpoints should call the ModelRegistry on app.state."""
    client, _state, _controller = hub_service_app

    class _StubRegistry:
        def __init__(self) -> None:
            """
            Initialize the tracker for loaded and unloaded models.

            Creates two empty lists, `loaded` and `unloaded`, used to record model names that have been loaded or unloaded during tests.
            """
            self.loaded: list[str] = []
            self.unloaded: list[str] = []

        async def request_vram_load(
            self, name: str, *, force: bool = False, timeout: float | None = None
        ) -> None:
            # Accept explicit `force`/`timeout` keyword args from the real API.
            # Reference them to avoid unused-variable warnings in static analysis.
            """
            Request loading a model into VRAM for testing purposes.

            Accepts `force` and `timeout` for compatibility with the real API (they are ignored).
            If `name` equals "denied", raises an HTTP 400 validation error with detail "denied".
            Otherwise records the model name by appending it to `self.loaded`.

            Parameters:
                name (str): The model name to load.
                force (bool): Ignored; kept for API compatibility.
                timeout (float | None): Ignored; kept for API compatibility.

            Raises:
                HTTPException: HTTP 400 with detail "denied" when `name` is "denied".
            """
            _ = (force, timeout)
            if name == "denied":
                # Simulate a validation error
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="denied")
            self.loaded.append(name)

        async def request_vram_unload(self, name: str, *, timeout: float | None = None) -> None:
            # Accept explicit `timeout` keyword arg from the real API and
            # reference it to avoid unused-variable warnings.
            """
            Record a VRAM unload request for the specified model name.

            Parameters:
                name (str): The model identifier to unload.
                timeout (float | None): Accepted for API compatibility but ignored.

            Raises:
                HTTPException: with status 400 and detail "not loaded" when `name` equals "missing".
            """
            _ = timeout
            if name == "missing":
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")
            self.unloaded.append(name)

    registry = _StubRegistry()
    client.app.state.model_registry = registry

    response = client.post("/hub/models/alpha/vram/load", json={})
    assert response.status_code == HTTPStatus.OK
    assert registry.loaded == ["alpha"]

    response = client.post("/hub/models/alpha/vram/unload", json={})
    assert response.status_code == HTTPStatus.OK
    assert registry.unloaded == ["alpha"]


def test_vram_admin_endpoints_surface_registry_errors(
    hub_service_app: tuple[TestClient, Any, Any],
) -> None:
    """Registry errors should propagate as HTTP responses from the VRAM endpoints."""
    client, _state, _controller = hub_service_app

    class _StubRegistryErr:
        async def request_vram_load(
            self, name: str, *, force: bool = False, timeout: float | None = None
        ) -> None:
            """
            Request loading the specified model's weights into VRAM.

            Parameters:
                name: The model identifier to load into VRAM.
                force (bool): If true, attempt to force the load despite capacity constraints.
                timeout (float | None): Maximum seconds to wait for the load to complete, or None to not wait.

            Raises:
                HTTPException: With status code 429 when the model's group is busy and the load cannot be accepted.
            """
            _ = (force, timeout)
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group busy")

        async def request_vram_unload(self, name: str, *, timeout: float | None = None) -> None:
            """
            Request unloading a model from VRAM.

            This implementation always fails when the model is not loaded and raises an HTTP 400 error with detail "not loaded".

            Parameters:
                name (str): Name of the model to unload from VRAM.
                timeout (float | None): Ignored; kept for API compatibility.

            Raises:
                HTTPException: Always raised with status code 400 and detail "not loaded".
            """
            _ = timeout
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")

    client.app.state.model_registry = _StubRegistryErr()

    response = client.post("/hub/models/denied/vram/load", json={})
    assert response.status_code == HTTPStatus.TOO_MANY_REQUESTS

    response = client.post("/hub/models/missing/vram/unload", json={})
    assert response.status_code == HTTPStatus.BAD_REQUEST


class FakeManager:
    """A minimal fake manager that simulates an unloaded handler."""

    async def unload(self, _reason: str) -> bool:
        """Simulate unloading the handler.

        Parameters
        ----------
        _reason : str
            Unused reason string (kept for compatibility with real managers).

        Returns
        -------
        bool
            Returns False to indicate there was no loaded handler.
        """
        return False

    def is_vram_loaded(self) -> bool:
        """
        Indicates whether VRAM is currently loaded for this manager.

        Returns:
            True if VRAM is loaded, False otherwise.
        """
        return False


@pytest.mark.asyncio
async def test_stop_model_removes_manager_and_updates_registry() -> None:
    """stop_model should clear the manager when unload() returns False.

    This ensures the supervisor no longer reports the model as running and
    that the registry is updated to reflect no attached handler.
    """
    hub_config = SimpleNamespace(
        models=[
            SimpleNamespace(
                name="m1",
                model_path="m1_path",
                is_default_model=False,
                jit_enabled=True,
                model_type="test",
                host="localhost",
                port=0,
                auto_unload_minutes=None,
            )
        ]
    )

    registry = ModelRegistry()
    # Register the model id used in the hub record so update_model_state can run
    registry.register_model("m1_path", handler=None, model_type="test")

    supervisor = HubSupervisor(hub_config, registry=registry)
    record = supervisor._models["m1"]
    # Attach a fake manager that will return False from unload()
    record.manager = FakeManager()

    result = await supervisor.stop_model("m1")

    assert result.get("status") == "stopped"
    assert record.manager is None
    # Registry handler should be None for the registered model id
    assert registry.get_handler("m1_path") is None


@pytest.mark.asyncio
async def test_hub_sync_once_updates_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `_hub_sync_once` reads hub snapshot and updates the registry."""
    # Prepare a fake snapshot returned by the hub daemon
    start_ts = 1764465807
    model_id = "mlx-community/Qwen3-30B-A3B-4bit-DWQ"
    snapshot = {
        "models": [
            {
                "name": "qwen3",
                "state": "running",
                "memory_loaded": True,
                "model_path": model_id,
                "started_at": start_ts,
                "active_requests": 2,
                "vram_load_error": None,
            }
        ]
    }

    async def fake_call(
        cfg: object, method: str, path: str, timeout: float = 2.0
    ) -> dict[str, Any]:
        """
        Return the pre-captured hub snapshot regardless of the supplied request parameters.

        Parameters:
            cfg: Ignored. Placeholder for a configuration object passed by callers.
            method: Ignored. HTTP method string (e.g., "GET", "POST").
            path: Ignored. Request path sent to the daemon API.
            timeout: Ignored. Timeout value for the call in seconds.

        Returns:
            dict: The hub snapshot dictionary provided by the enclosing test fixture.
        """
        return snapshot

    def fake_load_cfg(req: object) -> SimpleNamespace:
        # Return a simple object with host/port attributes
        """
        Produce a SimpleNamespace containing host and port for a fake configuration.

        Parameters:
            req (object): Ignored; present for compatibility with callers.

        Returns:
            SimpleNamespace: An object with attributes `host` (str) set to "127.0.0.1" and `port` (int) set to 5005.
        """
        return SimpleNamespace(host="127.0.0.1", port=5005)

    # Patch the functions as used by the server module (they are imported
    # into `app.server` at module load time), so override there.
    monkeypatch.setattr("app.server._call_daemon_api_async", fake_call)
    monkeypatch.setattr("app.server._load_hub_config_from_request", fake_load_cfg)

    app = FastAPI()
    registry = ModelRegistry()
    app.state.model_registry = registry

    # Register the model so update_model_state will accept it
    registry.register_model(model_id, handler=None, model_type="lm")

    # Run one sync iteration
    await _hub_sync_once(app)

    # Validate registry was updated
    status = registry.get_vram_status(model_id)
    assert status["vram_loaded"] is True
    extra = registry._extra[model_id]
    assert extra["vram_last_load_ts"] == start_ts
    assert extra["active_requests"] == 2
    assert extra.get("vram_load_error") is None
    assert extra.get("status") == "loaded"
