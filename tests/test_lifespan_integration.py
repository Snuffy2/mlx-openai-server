"""Integration tests for FastAPI lifespan/startup behavior.

These tests patch `instantiate_handler` to a lightweight fake and verify
that `create_lifespan` honors the `jit_enabled` setting (loads handler
immediately when JIT is disabled and defers when JIT is enabled).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

from app.config import MLXServerConfig
from app.server import create_lifespan


def test_lifespan_respects_jit_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that when JIT is disabled the handler is instantiated during FastAPI lifespan startup.

    During the app lifespan this test asserts the handler manager has a loaded handler and that the handler instantiation function is called exactly once.
    """
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        """
        Create and return a lightweight handler-like object for testing and increment a call counter.

        Parameters:
            cfg (MLXServerConfig): Server configuration whose `model_path` is propagated to the created object.

        Returns:
            object: A simple handler-like object with attributes:
                - model_path (str): copied from `cfg.model_path`.
                - model_created (int): Unix timestamp (seconds) when the object was created.

        Side effects:
            Increments called["count"] by 1 to track invocation count for tests.
        """
        called["count"] += 1
        # Return a simple handler-like object
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=False, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        """
        Run the FastAPI lifespan context and assert the handler manager loaded a handler during startup.

        While the lifespan is active, retrieve the app's handler manager and assert it has a current_handler and that the handler instantiation counter equals one.
        """
        async with lifespan(app):
            # Handler manager should have loaded the handler on startup
            hm = app.state.handler_manager
            assert hm.current_handler is not None
            assert called["count"] == 1

    asyncio.run(_run())


def test_lifespan_respects_jit_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When JIT is enabled the handler is not loaded at startup."""
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        """
        Create a lightweight fake handler-like object and record that instantiation was attempted.

        Parameters:
            cfg (MLXServerConfig): Server configuration whose `model_path` will be copied into the fake object.

        Returns:
            object: A SimpleNamespace with attributes `model_path` (from cfg) and `model_created` (an integer timestamp).
        Side effects:
            Increments `called["count"]` to signal that instantiation was invoked.
        """
        called["count"] += 1
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=True, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        """
        Run the app lifespan and verify that no handler is loaded at startup and no handler instantiation occurred.

        Opens the FastAPI lifespan for the test app, obtains the handler manager from app.state, and asserts that its `current_handler` is None and that the `called["count"]` counter remains zero.
        """
        async with lifespan(app):
            hm = app.state.handler_manager
            # No handler should be loaded at startup when JIT is enabled
            assert hm.current_handler is None
            assert called["count"] == 0

    asyncio.run(_run())


def test_lifespan_with_fake_handler_ensure_vram_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Integration test that verifies a handler loads VRAM during a handler_session and is cleaned up on shutdown when JIT is disabled.

    Patches `instantiate_handler` to return a FakeHandler that increments counters from `ensure_vram_loaded` and `cleanup`, runs the FastAPI lifespan with `jit_enabled=False`, waits for the registry to attach the handler, enters a `handler_session` to trigger VRAM loading, and asserts that cleanup is invoked after shutdown.
    """
    calls = {"ensure": 0, "cleanup": 0}

    class FakeHandler:
        def __init__(self, cfg: MLXServerConfig) -> None:
            """
            Initialize the handler and record the model path from the server configuration.

            Parameters:
                cfg (MLXServerConfig): Server configuration whose `model_path` will be stored on the handler as `model_path`.
            """
            self.model_path = cfg.model_path

        async def initialize(self, _cfg: MLXServerConfig) -> None:  # pragma: no cover - trivial
            """
            No-op initializer retained for interface compatibility.

            This coroutine intentionally ignores the provided configuration and performs no initialization.
            """
            return

        async def ensure_vram_loaded(
            self,
            *,
            _force: bool = False,
            _timeout: float | None = None,
        ) -> None:
            """
            Ensure the handler's model weights are loaded into VRAM.

            If `_force` is True, reload VRAM contents even if already loaded. If `_timeout` is provided, wait up to that many seconds for loading to complete; `None` means wait indefinitely.

            Parameters:
                _force (bool): Force reloading VRAM if True.
                _timeout (float | None): Maximum time in seconds to wait for loading, or `None` for no timeout.
            """
            await asyncio.sleep(0)
            calls["ensure"] += 1

        async def cleanup(self) -> None:
            """
            Perform cleanup actions for the handler and yield control to the event loop.

            Increments the shared `calls["cleanup"]` counter to record that cleanup occurred and awaits briefly to allow other pending tasks to run.
            """
            await asyncio.sleep(0)
            calls["cleanup"] += 1

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        """
        Create and return a FakeHandler instance configured with the provided server config.

        Parameters:
            cfg (MLXServerConfig): Server configuration used to initialize the fake handler.

        Returns:
            FakeHandler: An instance of FakeHandler initialized with `cfg`.
        """
        return FakeHandler(cfg)

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=False, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        """
        Run the application's lifespan, wait for the model's handler to attach, enter a handler session to trigger VRAM loading, and verify that handler cleanup occurs after shutdown.

        This function:
        - Starts the FastAPI lifespan context for `app`.
        - Locates a model id from the registry and waits briefly until a handler is attached for that model.
        - Enters a handler session for that model to ensure `ensure_vram_loaded` is invoked at least once.
        - Exits the lifespan so that handler cleanup can run and be observed by the test.
        """
        async with lifespan(app):
            registry = app.state.model_registry
            registry_model_id = next(m["id"] for m in registry.list_models())

            # Wait until the registry has attached the handler (update happens
            # asynchronously from the handler_manager on_change callback).
            for _ in range(20):
                try:
                    h = registry.get_handler(registry_model_id)
                except KeyError:
                    h = None
                if h is not None:
                    break
                await asyncio.sleep(0.01)

            # Enter a handler_session which should call ensure_vram_loaded on the handler
            async with registry.handler_session(registry_model_id):
                # ensure_vram_loaded should have been invoked by the session
                assert calls["ensure"] >= 1

        # After exiting lifespan, cleanup should have been called

    asyncio.run(_run())
    assert calls["cleanup"] >= 1


def test_jit_triggers_handler_load_on_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that when JIT is enabled the handler manager loads a handler on the first request.

    This test ensures the app starts without a preloaded handler, that calling handler_manager.ensure_loaded simulates a first request and causes the handler to be instantiated exactly once.
    """
    called = {"count": 0}

    async def fake_instantiate(cfg: MLXServerConfig) -> object:
        """
        Create a lightweight fake handler-like object and record that instantiation was attempted.

        Parameters:
            cfg (MLXServerConfig): Server configuration whose `model_path` will be copied into the fake object.

        Returns:
            object: A SimpleNamespace with attributes `model_path` (from cfg) and `model_created` (an integer timestamp).
        Side effects:
            Increments `called["count"]` to signal that instantiation was invoked.
        """
        called["count"] += 1
        return SimpleNamespace(model_path=cfg.model_path, model_created=int(time.time()))

    monkeypatch.setattr("app.server.instantiate_handler", fake_instantiate)

    config = MLXServerConfig(model_path="/models/test", jit_enabled=True, auto_unload_minutes=None)
    app = FastAPI()
    lifespan = create_lifespan(config)

    async def _run() -> None:
        """
        Exercise the FastAPI lifespan to verify on-demand handler loading when JIT is enabled.

        Runs the application's lifespan, grabs the handler manager, asserts no handler is preloaded, calls `ensure_loaded` to simulate a first request, and asserts a handler is returned and the instantiate routine was invoked exactly once.
        """
        async with lifespan(app):
            hm = app.state.handler_manager
            # Initially not loaded under JIT
            assert hm.current_handler is None

            # Trigger load via ensure_loaded (simulates first request)
            handler = await hm.ensure_loaded("test-request")
            assert handler is not None
            assert called["count"] == 1

    asyncio.run(_run())
