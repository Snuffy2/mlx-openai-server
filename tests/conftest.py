"""Shared test fixtures and helpers for the test suite.

This module provides small factory fixtures used by multiple test modules
to avoid duplication of common test helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
import pytest

from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig
from app.server import LazyHandlerManager


@pytest.fixture
def make_hub_config(tmp_path: Path) -> Callable[..., MLXHubConfig]:
    """
    Create a factory that builds MLXHubConfig instances with overridable fields.
    
    The returned callable accepts keyword overrides for the following fields:
    `host`, `port`, `model_starting_port`, `log_level`, `log_path`, `enable_status_page`,
    `source_path`, and `models`. If an override is not provided, these defaults are used:
    host='127.0.0.1', port=8123, model_starting_port=8000, log_level='INFO',
    log_path=tmp_path / 'logs', enable_status_page=True, source_path=tmp_path / 'hub.yaml'.
    If `models` is not provided, the default is a single LM model at `/models/foo` named "foo".
    
    Parameters:
        tmp_path (Path): Base temporary directory used to construct default `log_path` and `source_path`.
    
    Returns:
        Callable[..., MLXHubConfig]: A callable that accepts keyword overrides and returns an MLXHubConfig.
    """

    def _factory(**overrides: object) -> MLXHubConfig:
        """
        Builds an MLXHubConfig with sensible defaults and optional overrides for common fields.
        
        Accepted override keys and defaults:
        - host: "127.0.0.1"
        - port: 8123
        - model_starting_port: 8000
        - log_level: "INFO"
        - log_path: tmp_path / "logs"
        - enable_status_page: True
        - source_path: tmp_path / "hub.yaml"
        - models: list of model configs (default is a single LM model at "/models/foo" named "foo")
        
        Returns:
            MLXHubConfig: A hub configuration populated with the provided overrides and models.
        """
        cfg = MLXHubConfig(
            host=overrides.get("host", "127.0.0.1"),
            port=overrides.get("port", 8123),
            model_starting_port=overrides.get("model_starting_port", 8000),
            log_level=overrides.get("log_level", "INFO"),
            log_path=overrides.get("log_path", tmp_path / "logs"),
            enable_status_page=overrides.get("enable_status_page", True),
            source_path=overrides.get("source_path", tmp_path / "hub.yaml"),
        )
        models = overrides.get(
            "models",
            [MLXServerConfig(model_path="/models/foo", name="foo", model_type="lm")],
        )
        cfg.models = models
        return cfg

    return _factory


@pytest.fixture
def hub_config_with_defaults(
    make_hub_config: Callable[..., MLXHubConfig], make_model_mock: Callable[..., MagicMock]
) -> MLXHubConfig:
    """
    Create an MLXHubConfig containing a default model and a regular model for lifecycle tests.
    
    Both models are created using the provided make_model_mock factory to keep test model mocks consistent.
    
    Returns:
        MLXHubConfig: Hub configuration containing the two mock models (a default model and a regular model).
    """
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

    return make_hub_config(models=[default_model, regular_model])


@pytest.fixture
def mock_handler_manager() -> MagicMock:
    """
    Create a MagicMock that simulates a LazyHandlerManager for tests.
    
    The mock is preconfigured with observable behaviour commonly used in tests:
    - is_vram_loaded() returns False
    - ensure_loaded is an AsyncMock
    - unload is an AsyncMock that returns True
    
    Returns:
        MagicMock: A mock object shaped like `LazyHandlerManager` with the above behaviours.
    """
    manager = MagicMock(spec=LazyHandlerManager)
    manager.is_vram_loaded.return_value = False
    manager.ensure_loaded = AsyncMock()
    manager.unload = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def hub_config_file(tmp_path: Path) -> Path:
    """
    Write a minimal hub.yaml used by CLI tests containing a log_path and a single model named "alpha".
    
    Returns:
        Path: Path to the written hub.yaml file.
    """
    config = tmp_path / "hub.yaml"
    log_dir = tmp_path / "logs"
    config.write_text(
        f"""
log_path: {log_dir}
models:
  - name: alpha
    model_path: /models/a
    model_type: lm
""".strip(),
    )
    return config


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[[], MLXHubConfig]:
    """
    Create a zero-argument factory that builds a simple MLXHubConfig for tests.
    
    The returned callable captures the provided tmp_path and, when invoked,
    produces an MLXHubConfig whose log_path is tmp_path / "logs" and which
    contains a single LM model named "foo" with model_path "/models/foo".
    
    Parameters:
    	tmp_path (Path): Temporary directory used as the base for the config's log_path.
    
    Returns:
    	factory (Callable[[], MLXHubConfig]): A callable that returns the described MLXHubConfig.
    """

    def _make_config() -> MLXHubConfig:
        """
        Create a minimal MLXHubConfig for tests with a single LM model.
        
        Returns:
            config (MLXHubConfig): Configuration with `log_path` set to the test temporary path's "logs"
            directory and `models` containing one MLXServerConfig for model "foo" (model_type "lm",
            model_path "/models/foo").
        """
        return MLXHubConfig(
            log_path=tmp_path / "logs",
            models=[MLXServerConfig(model_path="/models/foo", name="foo", model_type="lm")],
        )

    return _make_config


@pytest.fixture
def live_snapshot() -> Callable[[int], dict]:
    """
    Create a generator function that produces a live snapshot payload for tests.
    
    The returned callable accepts a single argument `pid` (default 4321) and returns a dictionary representing a hub snapshot. The snapshot contains a "models" list with one model having keys: "name", "state", "pid", "log_path", and "started_at".
    
    Returns:
        fn (Callable[[int], dict]): A function that when called with `pid` returns the snapshot dictionary.
    """

    def _live_snapshot(pid: int = 4321) -> dict:
        """
        Create a minimal live snapshot payload containing a single running model.
        
        Parameters:
            pid (int): Process ID to include for the model snapshot.
        
        Returns:
            dict: Snapshot with a "models" list containing one model object with keys:
                - "name": model name
                - "state": model lifecycle state
                - "pid": process ID (from `pid`)
                - "log_path": path to the model log
                - "started_at": timestamp when the model started
        """
        return {
            "models": [
                {
                    "name": "foo",
                    "state": "running",
                    "pid": pid,
                    "log_path": "/tmp/foo.log",
                    "started_at": 1,
                },
            ],
        }

    return _live_snapshot


@pytest.fixture
def write_hub_yaml(tmp_path: Path) -> Callable[[str, str], Path]:
    """
    Create a writer function that writes given YAML content into a file under the provided temporary path.
    
    Parameters:
        tmp_path (Path): Directory used as the base path for created files.
    
    Returns:
        Callable[[str, str], Path]: A function that accepts (content, filename) — writes `content` (trimmed) to `tmp_path/filename` and returns the file Path.
    """

    def _write(content: str, filename: str = "hub.yaml") -> Path:
        """
        Write provided text to a file located in the module's temporary test path.
        
        Parameters:
            content (str): Text to write to the file; leading and trailing whitespace will be stripped.
            filename (str): Name of the file to create under the temporary path (default "hub.yaml").
        
        Returns:
            Path: Path object pointing to the created file.
        """
        p = tmp_path / filename
        p.write_text(content.strip())
        return p

    return _write


@pytest.fixture
def make_model_mock() -> Callable[..., MagicMock]:
    """
    Create a factory that produces configurable MagicMock objects representing a model.
    
    The returned callable has signature
    `(name: str = "mock_model", model_path: str = "/models/mock", model_type: str = "lm", *, is_default: bool = False, group: str | None = None, auto_unload_minutes: int | None = None) -> MagicMock`.
    Parameters:
        name (str): Model name.
        model_path (str): Filesystem path or identifier for the model.
        model_type (str): Model type identifier (e.g., "lm").
        is_default (bool): Whether the model should be marked as the default model.
        group (str | None): Optional model group name.
        auto_unload_minutes (int | None): Optional automatic unload timeout in minutes.
    
    Returns:
        MagicMock: A mock object with attributes `name`, `model_path`, `model_type`, `is_default_model`, `group`, and `auto_unload_minutes` set from the factory arguments.
    """

    def _factory(
        name: str = "mock_model",
        model_path: str = "/models/mock",
        model_type: str = "lm",
        *,
        is_default: bool = False,
        group: str | None = None,
        auto_unload_minutes: int | None = None,
    ) -> MagicMock:
        """
        Create a MagicMock that simulates a model object with common model attributes.
        
        Parameters:
            name (str): Model name.
            model_path (str): Filesystem or repository path for the model.
            model_type (str): Type/category of the model (e.g., "lm").
            is_default (bool): Whether this model is marked as the default model.
            group (str | None): Optional group identifier the model belongs to.
            auto_unload_minutes (int | None): Optional auto-unload timeout in minutes.
        
        Returns:
            MagicMock: A mock object with attributes `name`, `model_path`, `model_type`, `is_default_model`, `group`, and `auto_unload_minutes` set to the provided values.
        """
        m = MagicMock()
        m.name = name
        m.model_path = model_path
        m.model_type = model_type
        m.is_default_model = is_default
        m.group = group
        m.auto_unload_minutes = auto_unload_minutes
        return m

    return _factory


class _StubServiceState:
    """Stub state for testing hub service interactions."""

    def __init__(self) -> None:
        """
        Initialize a fresh stub service state used by tests.
        
        Attributes:
            available (bool): Whether the service is considered available.
            reload_calls (int): Number of times reload was invoked.
            reload_result (dict[str, list[str]]): Result placeholder for reload containing lists for "started", "stopped", and "unchanged".
            status_payload (dict[str, Any]): Status payload including "models" and "timestamp".
            start_calls (list[str]): Names of models requested to start.
            stop_calls (list[str]): Names of models requested to stop.
            shutdown_called (bool): True if shutdown was requested.
            controller_stop_calls (int): Number of times the controller stop was invoked.
        """
        self.available = True
        self.reload_calls = 0
        self.reload_result: dict[str, list[str]] = {"started": [], "stopped": [], "unchanged": []}
        self.status_payload: dict[str, Any] = {"models": [], "timestamp": 1}
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.shutdown_called = False
        self.controller_stop_calls = 0


class _StubController:
    """Stub controller for testing hub controller interactions."""

    def __init__(self) -> None:
        """
        Initialize the stub controller's tracking state for model operations.
        
        Attributes:
            loaded (list[str]): Names of models that have been loaded.
            unloaded (list[str]): Names of models that have been unloaded.
            started (list[str]): Names of models that have been started.
            stopped (list[str]): Names of models that have been stopped.
            reload_count (int): Number of times the controller's configuration has been reloaded.
        """
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reload_count = 0

    async def load_model(self, name: str) -> None:
        """
        Record that a model was loaded for the stub controller and simulate a quota error for a denied model.
        
        Parameters:
            name (str): The model name to mark as loaded.
        
        Raises:
            HTTPException: If `name` is "denied", raises an HTTPException with status 429 and detail "group busy".
        """
        self.loaded.append(name)
        if name == "denied":
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group busy")

    async def unload_model(self, name: str) -> None:
        """
        Record that a model unload was requested for the given model name.
        
        Parameters:
            name (str): The model name to unload.
        
        Raises:
            HTTPException: If `name` equals "missing", raises with status 400 and detail "not loaded".
        """
        self.unloaded.append(name)
        if name == "missing":
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")

    async def start_model(self, name: str) -> None:
        """
        Record a model start request and simulate a saturation error when the model name is "saturated".
        
        Raises:
            HTTPException: with status code 429 and detail "group full" when `name` is "saturated".
        """
        self.started.append(name)
        if name == "saturated":
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group full")

    async def stop_model(self, name: str) -> None:
        """
        Record that a model stop was requested by appending its name to the internal `stopped` list.
        
        Parameters:
            name (str): The name of the model to stop.
        """
        self.stopped.append(name)

    async def reload_config(self) -> dict[str, Any]:
        """
        Produce a reload result payload and record that a reload was performed.
        
        Increments the instance's `reload_count` and returns a dictionary describing which models were started, stopped, or left unchanged.
        
        Returns:
            dict[str, Any]: A dictionary with three keys:
                - 'started': list of model names that were started.
                - 'stopped': list of model names that were stopped.
                - 'unchanged': list of model names that were left unchanged.
        """
        self.reload_count += 1
        return {"started": [], "stopped": ["old_model"], "unchanged": ["alpha", "beta"]}

    def get_status(self) -> dict[str, Any]:
        """
        Provide a static status snapshot containing two example models ("alpha" and "beta") and their runtime metadata.
        
        Returns:
            dict: A status payload with keys:
                - "timestamp" (float): Epoch timestamp for the snapshot.
                - "models" (list[dict]): Two model status dictionaries. Each model dict contains:
                    - "name" (str): Model identifier.
                    - "state" (str | None): Model lifecycle state (e.g., "running", "stopped").
                    - "pid" (int | None): Process ID if running, otherwise `None`.
                    - "port" (int | None): Assigned port for the model.
                    - "started_at" (float | None): Epoch time when the model started, or `None`.
                    - "exit_code" (int | None): Process exit code if available, otherwise `None`.
                    - "memory_loaded" (bool): Whether the model is loaded in memory.
                    - "group" (str | None): Optional group name.
                    - "is_default" (bool): Whether the model is the default model.
                    - "model_path" (str): Filesystem path for the model.
                    - "auto_unload_minutes" (int | None): Auto-unload timeout in minutes, or `None`.
        """
        return {
            "timestamp": 1234567890.0,
            "models": [
                {
                    "name": "alpha",
                    "state": "running",
                    "pid": 12345,
                    "port": 8124,
                    "started_at": 1234567800.0,
                    "exit_code": None,
                    "memory_loaded": True,
                    "group": None,
                    "is_default": True,
                    "model_path": "test/path",
                    "auto_unload_minutes": 120,
                },
                {
                    "name": "beta",
                    "state": "stopped",
                    "pid": None,
                    "port": 8125,
                    "started_at": None,
                    "exit_code": None,
                    "memory_loaded": False,
                    "group": None,
                    "is_default": False,
                    "model_path": "test/path2",
                    "auto_unload_minutes": None,
                },
            ],
        }


@pytest.fixture
def stub_service_state() -> _StubServiceState:
    """
    Provide a fresh _StubServiceState instance used by tests.
    
    Returns:
        _StubServiceState: A new stub object representing service state with default counters, call lists, and flags.
    """
    return _StubServiceState()


@pytest.fixture
def stub_controller() -> _StubController:
    """Return a fresh stub controller instance for tests."""
    return _StubController()


class _StubServiceClient:
    def __init__(self) -> None:
        """
        Initialize the stub service client used in tests.
        
        Attributes:
            started (list[str]): Names of models passed to start_model, in call order.
            stopped (list[str]): Names of models passed to stop_model, in call order.
            reload_calls (int): Number of times reload() was invoked.
            shutdown_called (bool): True if shutdown() has been called.
            is_available_calls (int): Number of times is_available() was called.
        """
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reload_calls = 0
        self.shutdown_called = False
        self.is_available_calls = 0

    def is_available(self) -> bool:
        """
        Indicates whether the stub service is available.
        
        Increments the `is_available_calls` counter each time it is invoked.
        
        Returns:
            `true` if the service is available, `false` otherwise.
        """
        self.is_available_calls += 1
        return True

    def start_model(self, name: str) -> None:
        """
        Record that a model was started by appending its name to the internal `started` list.
        
        Parameters:
            name (str): Name of the model to mark as started.
        """
        self.started.append(name)

    def stop_model(self, name: str) -> None:
        """
        Record that a model was stopped by appending its name to the instance's stopped list.
        
        Parameters:
            name (str): Name of the model to mark as stopped.
        """
        self.stopped.append(name)

    def reload(self) -> dict[str, list[str]]:
        """
        Record that a reload was requested and return a stubbed reload result.
        
        Returns:
            dict[str, list[str]]: Mapping with keys "started", "stopped", and "unchanged", each containing an empty list.
        """
        self.reload_calls += 1
        return {"started": [], "stopped": [], "unchanged": []}

    def shutdown(self) -> None:
        """
        Record that the client has been shut down by setting its internal flag.
        """
        self.shutdown_called = True


@pytest.fixture
def stub_service_client() -> _StubServiceClient:
    """Provide a fresh stub service client for CLI tests."""
    return _StubServiceClient()