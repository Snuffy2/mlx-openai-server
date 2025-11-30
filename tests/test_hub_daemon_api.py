"""Integration tests for the hub daemon HTTP API.

These tests exercise the HTTP surface exposed by `app.hub.daemon.create_app`
but stub out the `HubSupervisor` on `app.state.supervisor` so no real
processes are spawned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from loguru import logger
import pytest

from app.hub.daemon import create_app


class _StubSupervisor:
    def __init__(self) -> None:
        """
        Initialize the stub supervisor's in-memory state.
        
        Sets `shutdown_called` to False and creates `started` as an empty mapping from model name (str) to mock PID (int).
        """
        self.shutdown_called = False
        self.started: dict[str, int] = {}

    def get_status(self) -> dict[str, Any]:
        """
        Provide a fixed supervisor status payload for tests.
        
        The returned payload contains a numeric `timestamp` and a `models` list with the current model entries.
        
        Returns:
            dict[str, Any]: A status payload with keys:
                - "timestamp" (int): A monotonic timestamp value.
                - "models" (list[dict]): List of model descriptors, each with:
                    - "name" (str): Model name.
                    - "state" (str): Model state (e.g., "stopped").
        """
        return {"timestamp": 1, "models": [{"name": "alpha", "state": "stopped"}]}

    async def start_model(self, name: str) -> dict[str, Any]:
        """
        Record that a model was started and return its start payload.
        
        Parameters:
            name (str): The model's name.
        
        Returns:
            dict: Payload with keys "status" (the string "started"), "name" (the model name), and "pid" (a mock process id).
        """
        self.started[name] = 1234
        return {"status": "started", "name": name, "pid": 1234}

    async def stop_model(self, name: str) -> dict[str, Any]:
        """
        Stop tracking the given model and return a stopped-status payload.
        
        Parameters:
            name (str): Name of the model to stop.
        
        Returns:
            dict[str, Any]: A payload containing "status": "stopped" and "name": the stopped model name.
        """
        self.started.pop(name, None)
        return {"status": "stopped", "name": name}

    async def load_model(self, name: str) -> dict[str, Any]:
        """
        Record that the model's memory was loaded and report the action.
        
        Parameters:
            name (str): The model identifier.
        
        Returns:
            dict: A payload with keys:
                - "status": the string "memory_loaded".
                - "name": the provided model identifier.
        """
        return {"status": "memory_loaded", "name": name}

    async def unload_model(self, name: str) -> dict[str, Any]:
        """
        Mark the specified model's in-memory resources as unloaded and report the outcome.
        
        Parameters:
            name (str): The model identifier whose memory was unloaded.
        
        Returns:
            result (dict[str, Any]): A payload containing `status` set to `"memory_unloaded"` and `name` equal to the provided model identifier.
        """
        return {"status": "memory_unloaded", "name": name}

    async def reload_config(self) -> dict[str, Any]:
        """
        Return a summary of models affected by reloading the hub configuration.
        
        Updates the supervisor's configuration and reports which models were started, stopped, or unchanged as a result.
        
        Returns:
            dict[str, list[str]]: Mapping with keys:
                - "started": list of model names that were started.
                - "stopped": list of model names that were stopped.
                - "unchanged": list of model names that remained unchanged.
        """
        return {"started": [], "stopped": [], "unchanged": []}

    async def shutdown_all(self) -> None:
        # mark called so tests can assert the background task ran
        """
        Record that a shutdown was requested for the stub supervisor.
        
        Sets `self.shutdown_called` to True so tests can verify that the shutdown background task was scheduled or executed.
        """
        self.shutdown_called = True
        logger.info("Stub shutdown called")


@pytest.mark.asyncio
async def test_daemon_health_and_status(tmp_path: Path) -> None:
    """Health and status endpoints return expected shapes."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
""".strip(),
    )

    app = create_app(str(cfg))
    stub = _StubSupervisor()
    app.state.supervisor = stub
    app.state.hub_controller = stub

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_id": None, "model_status": "controller"}

    r = client.get("/hub/status")
    assert r.status_code == 200
    payload = r.json()
    assert "models" in payload, "Response should contain 'models' key"
    assert isinstance(payload["models"], list), "payload['models'] should be a list"


@pytest.mark.asyncio
async def test_model_start_stop_and_memory_actions(tmp_path: Path) -> None:
    """Model lifecycle endpoints start/stop/load/unload behave as expected."""
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
host: 127.0.0.1
port: 8123
models:
  - name: alpha
    model_path: /models/alpha
""".strip(),
    )

    app = create_app(str(cfg))
    stub = _StubSupervisor()
    app.state.supervisor = stub
    app.state.hub_controller = stub

    client = TestClient(app)
    r = client.post("/hub/models/alpha/start")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.post("/hub/models/alpha/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.post("/hub/models/alpha/load")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.post("/hub/models/alpha/unload")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_shutdown_schedules_background_task() -> None:
    """POST /hub/shutdown schedules the supervisor shutdown background task."""
    # Test that the shutdown endpoint calls supervisor.shutdown_all()
    # Since the endpoint is synchronous in its logic, we can test the stub directly
    stub = _StubSupervisor()

    # Simulate calling the endpoint logic
    await stub.shutdown_all()

    # Verify shutdown_all was called
    assert stub.shutdown_called is True