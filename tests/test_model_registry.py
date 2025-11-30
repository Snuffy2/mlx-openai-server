"""Unit tests for ModelRegistry behavior."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any

from app.core.model_registry import ModelRegistry


def test_registry_tracks_handlers_and_metadata() -> None:
    """Ensure the registry records status, handler references, and metadata."""
    asyncio.run(_exercise_registry())


async def _exercise_registry() -> None:
    """
    Exercise ModelRegistry lifecycle operations and assert expected state transitions.
    
    Registers a model, verifies its initial metadata and status, updates the model state to attach a handler and apply metadata updates while checking the handler and metadata changes, detaches the handler to verify it is cleared, and finally unregisters the model confirming the registry count goes to zero.
    """
    registry = ModelRegistry()
    registry.register_model(
        model_id="foo",
        handler=None,
        model_type="lm",
        context_length=2048,
        metadata_extras={"group": "default"},
    )

    models = registry.list_models()
    assert len(models) == 1
    first = models[0]
    assert first["id"] == "foo"
    assert first["metadata"]["status"] == "unloaded"
    assert first["metadata"]["group"] == "default"

    handler = object()
    await registry.update_model_state(
        "foo",
        handler=handler,
        status="initialized",
        metadata_updates={"model_path": "/models/foo"},
    )

    assert registry.get_handler("foo") is handler
    updated = registry.list_models()[0]
    assert updated["metadata"]["status"] == "initialized"
    assert updated["metadata"]["model_path"] == "/models/foo"

    await registry.update_model_state("foo", handler=None, status="unloaded")
    assert registry.get_handler("foo") is None

    await registry.unregister_model("foo")
    assert registry.get_model_count() == 0


class DummyManager:
    """A simple manager mock that records calls and simulates VRAM state."""

    def __init__(self) -> None:
        """
        Initialize a DummyManager that simulates VRAM loading state and tracks calls and active sessions.
        
        Sets:
        - _loaded: whether VRAM is currently considered loaded.
        - load_calls / unload_calls: counters for number of load and unload requests.
        - ensure_lock: asyncio.Lock used to serialize load/unload operations.
        - _active_requests: number of active sessions using the manager.
        """
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0
        self.ensure_lock = asyncio.Lock()
        self._active_requests = 0

    def is_vram_loaded(self) -> bool:
        """
        Indicates whether the manager currently has VRAM loaded.
        
        Returns:
            bool: True if VRAM is loaded, False otherwise.
        """
        return self._loaded

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:
        """
        Ensure the manager's VRAM is loaded, performing a load if necessary.
        
        Acquires the manager's internal lock to serialize concurrent callers. If a load is performed, increments `load_calls` and marks the manager as loaded; if the manager is already loaded and `force` is False, this call is a no-op.
        
        Parameters:
        	force (bool): If True, force a reload even if VRAM is already marked loaded.
        	timeout (float | None): Optional timeout value accepted for API compatibility; currently ignored by this implementation.
        """
        async with self.ensure_lock:
            # Simulate expensive load
            if not self._loaded or force:
                self.load_calls += 1
                await asyncio.sleep(0.01)
                self._loaded = True

    async def release_vram(self, *, timeout: float | None = None) -> None:
        """
        Release VRAM held by this manager.
        
        If VRAM is currently loaded, increments `unload_calls`, waits approximately 0.005 seconds to simulate an unload, and marks the manager as not loaded.
        
        Parameters:
        	timeout (float | None): Ignored by this mock implementation; present for API compatibility.
        """
        async with self.ensure_lock:
            if self._loaded:
                self.unload_calls += 1
                await asyncio.sleep(0.005)
                self._loaded = False

    def request_session(
        self,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[Any]:
        """
        Provide an async context manager for a request session tied to this manager.
        
        The context manager increments the manager's active request count for the duration of the session and, if requested, ensures the manager's VRAM is loaded before yielding.
        
        Parameters:
            ensure_vram (bool): If True, ensure VRAM is loaded before entering the session.
            ensure_timeout (float | None): Timeout in seconds to wait for VRAM to become available; None means no timeout.
        
        Returns:
            AbstractAsyncContextManager[Any]: An async context manager that yields the manager for use within the session.
        """
        return DummySession(self, ensure_vram, ensure_timeout)


class DummySession:
    """Simple async context manager for DummyManager sessions."""

    def __init__(
        self,
        manager: DummyManager,
        ensure_vram: bool,
        ensure_timeout: float | None,
    ) -> None:
        """
        Initialize the DummySession with its manager and VRAM ensure settings.
        
        Parameters:
            manager (DummyManager): Manager instance the session will operate on.
            ensure_vram (bool): If True, entering the session will call the manager to ensure VRAM is loaded.
            ensure_timeout (float | None): Timeout in seconds passed to the manager's ensure call; None means no timeout.
        """
        self.manager = manager
        self.ensure_vram = ensure_vram
        self.ensure_timeout = ensure_timeout

    async def __aenter__(self) -> DummyManager:
        """
        Enter the context, increment the manager's active request count, and optionally ensure the manager's VRAM is loaded.
        
        Returns:
            DummyManager: The underlying manager instance associated with this session.
        """
        self.manager._active_requests += 1
        if self.ensure_vram:
            await self.manager.ensure_vram_loaded(timeout=self.ensure_timeout)
        return self.manager

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Release the session and decrement the manager's active request count.
        
        Called on exiting the async context; decreases the associated manager's active request counter by one.
        """
        self.manager._active_requests -= 1


def test_get_or_attach_manager_concurrent() -> None:
    """Concurrent callers should share a single loader invocation (stress test).

    This test uses the same registry implementation but spawns multiple
    concurrent callers to `get_or_attach_manager` to ensure only one
    loader runs.
    """

    async def _test() -> None:
        registry = ModelRegistry()
        model_id = "concurrent-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        loader_call_count = 0

        async def loader(mid: str) -> DummyManager:
            """
            Create and return a new DummyManager for the given model id after a short simulated initialization delay.
            
            Parameters:
                mid (str): Model identifier for which the manager is being created.
            
            Returns:
                DummyManager: A newly constructed DummyManager instance.
            """
            nonlocal loader_call_count
            loader_call_count += 1
            # Simulate async init delay
            await asyncio.sleep(0.02)
            return DummyManager()

        # Spawn multiple concurrent callers that should share one loader task.
        tasks = [
            asyncio.create_task(registry.get_or_attach_manager(model_id, loader)) for _ in range(8)
        ]
        results = await asyncio.gather(*tasks)

        # All returned managers should be the same instance
        assert all(r is results[0] for r in results)
        assert loader_call_count == 1
        assert registry.get_handler(model_id) is results[0]

    asyncio.run(_test())


def test_request_vram_load_unload_idempotent_concurrent() -> None:
    """Concurrent load/unload calls should behave idempotently."""

    async def _test() -> None:
        registry = ModelRegistry()
        model_id = "vram-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        # Attach a DummyManager directly via update_model_state (simulate an attached manager)
        manager = DummyManager()
        await registry.update_model_state(model_id, handler=manager)

        # Concurrently request loads
        load_tasks = [asyncio.create_task(registry.request_vram_load(model_id)) for _ in range(6)]
        await asyncio.gather(*load_tasks)

        # Manager must report loaded and registry should reflect it
        status = registry.get_vram_status(model_id)
        assert status["vram_loaded"] is True
        assert status["vram_last_load_ts"] is not None
        assert manager.load_calls == 1

        # Concurrently request unloads
        unload_tasks = [
            asyncio.create_task(registry.request_vram_unload(model_id)) for _ in range(4)
        ]
        await asyncio.gather(*unload_tasks)

        status2 = registry.get_vram_status(model_id)
        assert status2["vram_loaded"] is False
        assert status2["vram_last_unload_ts"] is not None
        # After unload, a load then another load (force) should increment
        await registry.request_vram_load(model_id)
        assert manager.load_calls >= 2

    asyncio.run(_test())


def test_handler_session_updates_active_requests_and_notifies() -> None:
    """
    Verify that handler_session increments and decrements the active request count and notifies registered observers.
    
    Within a session, the model's VRAM status should report active_requests == 1; after the session, active_requests should be 0 and any registered notifier should have been called with the model id.
    """

    async def _test() -> None:
        """
        Verify that handler_session increments the model's active request count while open and notifies registered observers when the session exits.
        
        This test registers a model with a DummyManager, registers an activity notifier, enters an asynchronous handler_session for the model and asserts that `active_requests` equals 1 inside the session, then asserts `active_requests` returns to 0 after exiting and the notifier was invoked with the model id.
        """
        registry = ModelRegistry()
        model_id = "session-model"
        registry.register_model(model_id, handler=None, model_type="lm")

        manager = DummyManager()
        await registry.update_model_state(model_id, handler=manager)

        notified: list[str] = []

        def notifier(mid: str) -> None:
            """
            Record a model ID in the surrounding notifier list.
            
            Appends the provided model identifier to the `notified` list captured from the enclosing scope.
            
            Parameters:
                mid (str): The model identifier to record.
            """
            notified.append(mid)

        registry.register_activity_notifier(notifier)

        async with registry.handler_session(model_id):
            # inside session active_requests should be 1
            status = registry.get_vram_status(model_id)
            assert status["active_requests"] == 1

        # after exiting, active_requests should be 0 and notifier called
        status2 = registry.get_vram_status(model_id)
        assert status2["active_requests"] == 0
        assert model_id in notified

    asyncio.run(_test())


def test_request_vram_idempotency() -> None:
    """VRAM load/unload requests should be idempotent and respect `force`."""

    async def _test() -> None:
        registry = ModelRegistry()
        manager = DummyManager()
        registry.register_model(model_id="idm", handler=manager, model_type="lm")

        # First load
        await registry.request_vram_load("idm")
        assert manager.is_vram_loaded()
        assert manager.load_calls == 1

        # Second load without force should be no-op (manager may be idempotent)
        await registry.request_vram_load("idm")
        assert manager.load_calls == 1

        # Force reload triggers another load
        await registry.request_vram_load("idm", force=True)
        assert manager.load_calls == 2

        # Unload
        await registry.request_vram_unload("idm")
        assert not manager.is_vram_loaded()
        assert manager.unload_calls == 1

        # Double unload is a no-op
        await registry.request_vram_unload("idm")
        assert manager.unload_calls == 1

    asyncio.run(_test())


def test_handler_session_counts_and_ensure() -> None:
    """handler_session should increment active_requests and ensure VRAM loaded."""

    async def _test() -> None:
        registry = ModelRegistry()
        manager = DummyManager()
        registry.register_model(model_id="sess", handler=manager, model_type="lm")

        # Before any sessions
        status0 = registry.get_vram_status("sess")
        assert status0["active_requests"] == 0

        # Enter a session; active_requests increments and ensure_vram called
        async with registry.handler_session("sess") as m:
            assert m is manager
            status1 = registry.get_vram_status("sess")
            assert status1["active_requests"] == 1
            # ensure_vram should have loaded the model
            assert manager.is_vram_loaded()

        # After exit, active_requests returns to zero
        status2 = registry.get_vram_status("sess")
        assert status2["active_requests"] == 0

    asyncio.run(_test())