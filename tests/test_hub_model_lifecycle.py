"""Tests for hub model lifecycle management.

These tests verify that models start/stop, load/unload correctly,
and that default models auto-start on hub initialization.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from app.hub.config import MLXHubConfig
from app.hub.daemon import HubSupervisor, create_app
from app.server import LazyHandlerManager


class _MockHandlerManager:
    """Mock handler manager for testing."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the mock handler manager for a given model path and default state.

        Parameters:
            model_path (str): Filesystem path or identifier of the model associated with this handler.
        """
        self.model_path = model_path
        self._handler = None
        self._shutdown = False

    def is_vram_loaded(self) -> bool:
        """
        Check whether the model's handler is present in memory, indicating VRAM is loaded.

        Returns:
            `true` if a handler is present (VRAM is loaded), `false` otherwise.
        """
        return self._handler is not None

    async def ensure_loaded(self, reason: str = "request") -> Any:
        """
        Ensure a mock handler exists for this manager and return it.

        Parameters:
            reason (str): Optional human-readable reason for loading.

        Returns:
            Any: The handler object if loaded, otherwise `None`.
        """
        if not self._shutdown:
            self._handler = MagicMock()
            self._handler.model_path = self.model_path
        return self._handler

    async def unload(self, reason: str = "manual") -> bool:
        """
        Unload the current handler and mark the manager as not loaded.

        Parameters:
                reason (str): Optional textual reason for the unload action (defaults to "manual"). For this mock implementation the value is informational only.

        Returns:
                true if the manager is now unloaded (handler cleared).
        """
        self._handler = None
        return True


class _TestHubSupervisor(HubSupervisor):
    """Test version of HubSupervisor with mocked handler creation."""

    def __init__(self, hub_config: MLXHubConfig) -> None:
        # Don't call super().__init__ to avoid registry setup
        """
        Initialize a test-focused HubSupervisor shim and populate in-memory model records from the provided hub configuration.

        This constructor intentionally does not call the superclass initializer; it sets up minimal state required for tests: stores the given hub_config, disables registry use, creates an asyncio lock, a background task list, and a shutdown flag. For each model in hub_config.models it adds a MagicMock record keyed by model name; each record contains: name, config, group, is_default, model_path, auto_unload_minutes, manager (initially None), started_at (initially None), and exit_code (initially None).

        Parameters:
            hub_config (MLXHubConfig): Hub configuration whose `models` iterable will be used to create the in-memory model records. Each model may be an object with attributes `name`, `group`, `is_default_model`, `model_path`, and `auto_unload_minutes`; when an attribute is missing a reasonable default is used.
        """
        self.hub_config = hub_config
        self.registry = None
        self._models = {}
        self._lock = asyncio.Lock()
        self._bg_tasks = []
        self._shutdown = False

        # Populate model records
        for model in getattr(hub_config, "models", []):
            name = getattr(model, "name", None) or str(model)
            record = MagicMock()
            record.name = name
            record.config = model
            record.group = getattr(model, "group", None)
            record.is_default = getattr(model, "is_default_model", False)
            record.model_path = getattr(model, "model_path", None)
            record.auto_unload_minutes = getattr(model, "auto_unload_minutes", None)
            record.manager = None
            record.started_at = None
            record.exit_code = None
            self._models[name] = record


@pytest.fixture
def hub_config_with_defaults(
    tmp_path: Path, make_model_mock: Callable[..., MagicMock]
) -> MLXHubConfig:
    """
    Create an MLXHubConfig populated with one default model and one regular model.

    Parameters:
        tmp_path (Path): Temporary directory used as the hub's source_path and base for log_path.
        make_model_mock (Callable[..., MagicMock]): Factory that returns a mocked model config when called as
            make_model_mock(name, model_path, model_type, **kwargs).

    Returns:
        MLXHubConfig: Hub configuration whose `models` list contains:
            - a model named "default_model" with `is_default=True` and model_path "/models/default"
            - a model named "regular_model" with `is_default=False` and model_path "/models/regular"
    """
    config = MLXHubConfig(
        host="127.0.0.1",
        port=8123,
        model_starting_port=8000,
        log_level="INFO",
        log_path=tmp_path / "logs",
        enable_status_page=True,
        source_path=tmp_path / "hub.yaml",
    )

    # Mock model configs using the shared factory from tests.conftest
    default_model = make_model_mock(
        "default_model",
        "/models/default",
        "lm",
        is_default=True,
        group=None,
        auto_unload_minutes=120,
    )

    regular_model = make_model_mock(
        "regular_model",
        "/models/regular",
        "lm",
        is_default=False,
        group=None,
        auto_unload_minutes=120,
    )

    config.models = [default_model, regular_model]
    return config


@pytest.fixture
def mock_handler_manager() -> MagicMock:
    """
    Create a MagicMock that simulates a LazyHandlerManager for tests.

    The mock has `is_vram_loaded()` returning `False`, `ensure_loaded` as an AsyncMock, and `unload` as an AsyncMock that returns `True`.

    Returns:
        MagicMock: A mock LazyHandlerManager configured for testing.
    """
    manager = MagicMock(spec=LazyHandlerManager)
    manager.is_vram_loaded.return_value = False
    manager.ensure_loaded = AsyncMock()
    manager.unload = AsyncMock(return_value=True)
    return manager


@pytest.mark.asyncio
async def test_model_start_sets_manager_and_memory_loaded(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that starting a model creates a handler manager and sets memory_loaded."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Mock the handler manager creation
    with patch("app.hub.daemon.LazyHandlerManager") as mock_lhm:
        mock_manager = MagicMock()
        mock_manager.ensure_loaded = AsyncMock(return_value=MagicMock())
        mock_manager.is_vram_loaded.return_value = True
        mock_manager.jit_enabled = False  # regular_model is not JIT-enabled
        mock_lhm.return_value = mock_manager

        result = await supervisor.start_model("regular_model")

        assert result["status"] == "loaded"
        assert result["name"] == "regular_model"

        # Check that the record was updated
        record = supervisor._models["regular_model"]
        assert record.manager is not None
        assert record.manager.is_vram_loaded() is True

        # Verify LazyHandlerManager was created with correct config
        mock_lhm.assert_called_once()
        call_args = mock_lhm.call_args[0][0]  # First positional arg
        assert call_args.model_path == "/models/regular"


@pytest.mark.asyncio
async def test_model_start_already_loaded_returns_early(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that starting an already loaded model returns early without recreation."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)
    record = supervisor._models["regular_model"]
    record.manager = MagicMock()
    record.manager.is_vram_loaded.return_value = True

    result = await supervisor.start_model("regular_model")

    assert result["status"] == "already_loaded"
    assert result["name"] == "regular_model"


@pytest.mark.asyncio
async def test_model_stop_unloads_and_clears_state(hub_config_with_defaults: MLXHubConfig) -> None:
    """Test that stopping a model unloads it and clears the manager."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.unload = AsyncMock(return_value=True)
    record.manager = mock_manager

    result = await supervisor.stop_model("regular_model")

    assert result["status"] == "stopped"
    assert result["name"] == "regular_model"

    # Verify unload was called and manager was cleared
    mock_manager.unload.assert_called_once_with("stop")
    assert record.manager is None


@pytest.mark.asyncio
async def test_model_load_and_unload(hub_config_with_defaults: MLXHubConfig) -> None:
    """
    Verifies that a supervisor can load and then unload a non-default model.

    This test starts the regular model to ensure a handler manager is present, then calls
    load_model and asserts it reports the model as loaded, and calls unload_model and
    asserts it reports the model as unloaded and that the manager's unload method was
    invoked with the reason "unload".

    Parameters:
        hub_config_with_defaults (MLXHubConfig): Test hub configuration containing a default and a regular model.
    """
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # First start the model to initialize the manager
    with patch("app.hub.daemon.LazyHandlerManager") as mock_lhm:
        mock_manager = MagicMock()
        mock_manager.ensure_loaded = AsyncMock(return_value=MagicMock())
        mock_manager.jit_enabled = False  # regular_model is not JIT-enabled
        mock_lhm.return_value = mock_manager

        start_result = await supervisor.start_model("regular_model")
        assert start_result["status"] == "loaded"

    # Now test load
    mock_manager.ensure_loaded = AsyncMock(return_value=MagicMock())
    result = await supervisor.load_model("regular_model")
    assert result["status"] == "loaded"
    assert result["name"] == "regular_model"

    # Test unload
    mock_manager.unload = AsyncMock(return_value=True)

    result = await supervisor.unload_model("regular_model")
    assert result["status"] == "unloaded"
    assert result["name"] == "regular_model"

    # Verify unload was called
    mock_manager.unload.assert_called_once_with("unload")


@pytest.mark.asyncio
async def test_get_status_returns_correct_state(hub_config_with_defaults: MLXHubConfig) -> None:
    """Test that get_status returns correct running/stopped states."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Set up one loaded model
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.is_vram_loaded.return_value = True
    record.manager = mock_manager

    status = supervisor.get_status()

    assert "models" in status
    assert len(status["models"]) == 2

    # Find the models in the status
    regular_model_status = next(m for m in status["models"] if m["name"] == "regular_model")
    default_model_status = next(m for m in status["models"] if m["name"] == "default_model")

    assert regular_model_status["state"] == "running"
    assert default_model_status["state"] == "stopped"


@pytest.mark.asyncio
async def test_reload_config_preserves_loaded_models(
    hub_config_with_defaults: MLXHubConfig,
) -> None:
    """Test that reload_config preserves loaded model state for unchanged models."""
    supervisor = _TestHubSupervisor(hub_config_with_defaults)

    # Load a model before reload
    record = supervisor._models["regular_model"]
    mock_manager = MagicMock()
    mock_manager.is_vram_loaded.return_value = True
    record.manager = mock_manager
    record.started_at = 12345

    # Mock load_hub_config to return the same config
    with patch(
        "app.hub.daemon.load_hub_config",
        return_value=hub_config_with_defaults,
    ) as mock_load:
        # Reload config (should preserve the loaded state)
        result = await supervisor.reload_config()

        assert "started" in result
        assert "stopped" in result
        assert "unchanged" in result

        # Verify the loaded model was preserved
        record_after = supervisor._models["regular_model"]
        assert record_after.manager is mock_manager
        assert record_after.manager.is_vram_loaded() is True
        assert record_after.started_at == 12345

        mock_load.assert_called_once_with(hub_config_with_defaults.source_path)
    assert "unchanged" in result


@pytest.mark.asyncio
async def test_default_models_auto_start_during_lifespan(
    tmp_path: Path, write_hub_yaml: Callable[[str, str], Path]
) -> None:
    """
    Verify that models marked as default in the hub configuration are started automatically during the application's lifespan.

    Parameters:
        tmp_path (Path): Temporary filesystem path provided by the test runner.
        write_hub_yaml (Callable[[str, str], Path]): Fixture that writes the given YAML content to a file and returns its Path.
    """
    cfg = write_hub_yaml(
        """
host: 127.0.0.1
port: 8123
models:
  - name: default_model
    model_path: /models/default
    model_type: lm
    default: true
  - name: regular_model
    model_path: /models/regular
    model_type: lm
""",
    )

    app = create_app(str(cfg))

    # Mock the supervisor to track start calls
    original_supervisor = app.state.supervisor
    start_calls = []

    async def mock_start_model(name: str) -> dict[str, Any]:
        """
        Record that a model was requested to start and report it as loaded.

        Parameters:
            name (str): The model's name.

        Returns:
            dict[str, Any]: A dictionary with keys `"status"` set to `"loaded"` and `"name"` equal to the provided model name.
        """
        start_calls.append(name)
        return {"status": "loaded", "name": name}

    # Replace the supervisor's start_model method
    original_supervisor.start_model = mock_start_model

    # Use TestClient as context manager to trigger the lifespan
    with TestClient(app):
        # The lifespan should have auto-started the default model
        pass

    # Restore the original supervisor
    app.state.supervisor = original_supervisor

    # Assert that the default model was auto-started
    assert "default_model" in start_calls
    assert "regular_model" not in start_calls


@pytest.mark.asyncio
async def test_hub_api_endpoints_call_supervisor_methods(
    tmp_path: Path, write_hub_yaml: Callable[[str, str], Path]
) -> None:
    """
    Verify hub HTTP endpoints invoke the supervisor's start, stop, load, and unload methods with the correct model name.

    Parameters:
        tmp_path (Path): Temporary directory fixture provided by pytest.
        write_hub_yaml (Callable[[str, str], Path]): Fixture that writes the given YAML content to a config file and returns its Path.
    """
    cfg = write_hub_yaml(
        """
host: 127.0.0.1
port: 8123
models:
  - name: test_model
    model_path: /models/test
    model_type: lm
""",
    )

    app = create_app(str(cfg))

    # Mock supervisor methods
    supervisor = app.state.supervisor
    supervisor.start_model = AsyncMock(return_value={"status": "loaded", "name": "test_model"})
    supervisor.stop_model = AsyncMock(return_value={"status": "stopped", "name": "test_model"})
    supervisor.load_model = AsyncMock(return_value={"status": "loaded", "name": "test_model"})
    supervisor.unload_model = AsyncMock(return_value={"status": "unloaded", "name": "test_model"})

    client = TestClient(app)

    # Test start
    response = client.post("/hub/models/test_model/start")
    assert response.status_code == 200
    supervisor.start_model.assert_called_with("test_model")

    # Test stop
    response = client.post("/hub/models/test_model/stop")
    assert response.status_code == 200
    supervisor.stop_model.assert_called_with("test_model")

    # Test load
    response = client.post("/hub/models/test_model/load")
    assert response.status_code == 200
    supervisor.load_model.assert_called_with("test_model")

    # Test unload
    response = client.post("/hub/models/test_model/unload")
    assert response.status_code == 200
    supervisor.unload_model.assert_called_with("test_model")
