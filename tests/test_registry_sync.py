"""Regression tests: ensure ModelRegistry stays in sync with HubSupervisor.

These tests verify that starting, stopping, loading and unloading models
update the registry handler attachment/status as expected.
"""

import pytest

from app.config import MLXServerConfig
from app.core.model_registry import ModelRegistry
from app.hub.config import MLXHubConfig
from app.hub.daemon import HubSupervisor


class StubManager:
    """Lightweight stub implementing the minimal manager protocol used by tests.

    The stub provides `is_vram_loaded`, `ensure_loaded`, `unload`, and
    `remove_log_sink` so it can be attached to `HubSupervisor` records
    without initializing real MLX handlers.
    """

    def __init__(self) -> None:
        """
        Initialize the StubManager.
        
        Sets the internal loaded flag to False indicating no model is currently loaded.
        """
        self._loaded = False

    def is_vram_loaded(self) -> bool:
        """
        Indicates whether the stub manager is currently loaded.
        
        Returns:
            bool: True if the stub is loaded, False otherwise.
        """
        return self._loaded

    async def ensure_loaded(self, _reason: str = "test") -> object:
        """
        Mark this stub as loaded and provide a dummy handler object.
        
        Returns:
            object: A new dummy handler object representing the loaded handler.
        """
        self._loaded = True
        return object()

    async def unload(self, _reason: str = "test") -> bool:
        """
        Unload the manager if it is currently loaded.
        
        Parameters:
            _reason (str): Optional reason for unloading; ignored by this stub.
        
        Returns:
            bool: `True` if the manager was loaded and was unloaded, `False` otherwise.
        """
        if self._loaded:
            self._loaded = False
            return True
        return False

    def remove_log_sink(self) -> None:
        """
        Placeholder that does nothing when asked to remove per-model log sinks.
        
        Present to satisfy the manager protocol; has no effect.
        """
        # Explicitly returning None is unnecessary.


@pytest.mark.asyncio
async def test_registry_updates_on_start_stop_load_unload() -> None:
    """Ensure registry handler attachment is updated after lifecycle actions."""

    cfg = MLXServerConfig(model_path="modelA", name="modelA", jit_enabled=True)
    hub_cfg = MLXHubConfig(models=[cfg])
    registry = ModelRegistry()
    # Register model in registry using the model_path as identifier
    registry.register_model(cfg.model_path, None, cfg.model_type)

    supervisor = HubSupervisor(hub_cfg, registry)

    # Start model should attach a manager in JIT mode
    await supervisor.start_model(cfg.name)
    assert registry.get_handler(cfg.model_path) is not None

    # Stop model should clear the manager and detach from registry
    await supervisor.stop_model(cfg.name)
    assert registry.get_handler(cfg.model_path) is None

    # Attach a stub manager and load the model via supervisor
    stub = StubManager()
    supervisor._models[cfg.name].manager = stub
    await supervisor.load_model(cfg.name)
    assert registry.get_handler(cfg.model_path) is stub

    # Unload should remove the manager and update registry
    await supervisor.unload_model(cfg.name)
    assert registry.get_handler(cfg.model_path) is None