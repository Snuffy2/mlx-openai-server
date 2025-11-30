"""Tests for the CentralIdleAutoUnloadController behavior.

This module contains both higher-level integration-style tests that use the
real `ModelRegistry` and lightweight unit-style tests that mock registry
responses. They were merged to keep controller behavior tests in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import time

import pytest

from app.core.model_registry import ModelRegistry
from app.server import CentralIdleAutoUnloadController


async def _wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: float = 1.0,
    poll_interval: float = 0.01,
) -> bool:
    """
    Waits until the provided condition returns True or the timeout is reached.

    Periodically evaluates condition_func every poll_interval seconds until it returns True or timeout seconds have elapsed.

    Parameters:
        condition_func (Callable[[], bool]): Function evaluated repeatedly; should return True when the desired condition is met.
        timeout (float): Maximum number of seconds to wait for the condition.
        poll_interval (float): Seconds to wait between successive evaluations of condition_func.

    Returns:
        bool: `True` if the condition became true before timeout, `False` otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(poll_interval)
    return False


class FakeManager:  # type: ignore[override]
    """Simple fake manager that tracks unload calls and VRAM state."""

    def __init__(self) -> None:
        """
        Create a manager initialized to report VRAM as loaded and with no recorded unload attempts.

        Attributes:
            _loaded (bool): True if VRAM is currently considered loaded.
            unload_calls (int): Number of times an unload has been attempted.
        """
        self._loaded = True
        self.unload_calls = 0

    def is_vram_loaded(self) -> bool:
        """
        Indicates whether the manager's VRAM is currently loaded.

        Returns:
            `True` if VRAM is loaded, `False` otherwise.
        """
        return self._loaded

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        _timeout: float | None = None,
    ) -> None:
        """
        Ensure the manager's VRAM state is marked as loaded.

        Parameters:
                force (bool): If True, mark VRAM as loaded even if it is already marked loaded. The method is idempotent.
        """
        await asyncio.sleep(0)
        if not self._loaded or force:
            self._loaded = True

    async def release_vram(self, *, _timeout: float | None = None) -> None:
        """
        Release simulated VRAM for this fake manager and record the unload.

        Marks the manager as not having VRAM loaded and increments the `unload_calls` counter if VRAM was previously loaded. Accepts an optional `_timeout` parameter which is ignored. The method yields briefly to allow cooperative scheduling.
        """
        await asyncio.sleep(0)
        if self._loaded:
            self.unload_calls += 1
            self._loaded = False


def test_central_controller_unloads_idle_model() -> None:
    """Controller should unload a model marked with immediate auto-unload."""

    async def _test() -> None:
        registry = ModelRegistry()
        mgr = FakeManager()

        # Register model with metadata that indicates immediate auto-unload
        registry.register_model(  # type: ignore[arg-type]
            model_id="m1",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # Wait for the controller to unload the model
        unloaded = await _wait_for_condition(
            lambda: not registry.get_vram_status("m1")["vram_loaded"],
        )
        assert unloaded, "Model should have been unloaded by the controller"

        status = registry.get_vram_status("m1")
        assert status["vram_loaded"] is False
        assert mgr.unload_calls >= 1

        await controller.stop()

    asyncio.run(_test())


def test_activity_resets_idle_timer() -> None:
    """Activity during a session prevents the central controller from unloading."""

    async def _test() -> None:
        """
        Verifies that activity during a handler session prevents auto-unload and that the model is unloaded after the session ends.

        Starts a controller with a model configured for immediate auto-unload (auto_unload_minutes = 0), opens a handler session for that model to simulate activity, asserts the model remains loaded while the session is active, then asserts the controller unloads the model promptly after the session closes.
        """
        registry = ModelRegistry()
        mgr = FakeManager()

        registry.register_model(  # type: ignore[arg-type]
            model_id="m2",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # While a session is active, the controller must not unload the model
        async with registry.handler_session("m2") as _m:
            # Give a moment for any potential unload to occur (it shouldn't)
            await asyncio.sleep(0.05)
            assert mgr.is_vram_loaded() is True

        # After session ends, controller should unload promptly
        unloaded = await _wait_for_condition(
            lambda: not registry.get_vram_status("m2")["vram_loaded"],
        )
        assert unloaded, "Model should have been unloaded after session ended"

        status = registry.get_vram_status("m2")
        assert status["vram_loaded"] is False

        await controller.stop()

    asyncio.run(_test())


def test_unload_failure_triggers_backoff() -> None:
    """
    Verify that when a model's unload operation fails, the controller applies a backoff and does not retry immediately.

    Registers a model whose manager raises on unload, starts the controller, asserts that exactly one unload attempt occurs, then waits to confirm no additional attempts are made during the backoff period.
    """

    async def _test() -> None:
        """
        Verifies the controller applies backoff when a model's unload fails and does not retry immediately.

        Registers a manager that raises on the first unload attempt, starts the CentralIdleAutoUnloadController, and asserts the controller attempts exactly one unload and does not retry during the backoff window before stopping the controller.
        """
        calls = {"count": 0}

        class FailManager(FakeManager):
            async def release_vram(self, *, timeout: float | None = None) -> None:
                """
                Simulate an attempted VRAM release that always fails.

                Increments the internal failure counter and then raises a RuntimeError to emulate an unload failure.

                Parameters:
                    timeout (float | None): Optional maximum time in seconds allowed for the unload attempt; accepted but not used by this simulation.

                Raises:
                    RuntimeError: Always raised to indicate a simulated unload failure.
                """
                calls["count"] += 1
                raise RuntimeError("simulated unload failure")

        registry = ModelRegistry()
        mgr = FailManager()

        registry.register_model(  # type: ignore[arg-type]
            model_id="m3",
            handler=mgr,
            model_type="lm",
            metadata_extras={"auto_unload_minutes": 0},
        )

        controller = CentralIdleAutoUnloadController(registry)
        registry.register_activity_notifier(controller.notify_activity)
        controller.start()

        # Wait for the controller to attempt unload once (should fail)
        first_attempt = await _wait_for_condition(lambda: calls["count"] >= 1)
        assert first_attempt, "Controller should have attempted to unload the model"
        assert calls["count"] == 1

        # Short wait to ensure no immediate retry (backoff applied)
        # Wait a bit longer than the backoff period to ensure no retry
        await asyncio.sleep(0.2)
        assert calls["count"] == 1, "No retry should occur during backoff period"

        await controller.stop()

    asyncio.run(_test())


@pytest.mark.asyncio
async def test_central_controller_unloads_idle_model_with_fake_registry() -> None:
    """Central controller should call registry.request_vram_unload for idle models."""
    unloaded = asyncio.Event()

    class FakeRegistry:
        def list_models(self) -> list[dict]:
            """
            List models known to the registry along with their metadata.

            Returns:
                A list of model dictionaries where each item contains an "id" (model identifier) and a "metadata" dictionary (for example, including "auto_unload_minutes").
            """
            return [{"id": "m1", "metadata": {"auto_unload_minutes": 0}}]

        def get_vram_status(self, _model_id: str) -> dict:
            """
            Provide a simulated VRAM status for the given model.

            Parameters:
                _model_id (str): Model identifier (ignored by this fake implementation).

            Returns:
                dict: A mapping with keys:
                    - "vram_loaded": `True` indicating VRAM is currently loaded.
                    - "vram_last_load_ts": UNIX timestamp of the last load (set to one hour ago).
                    - "vram_last_unload_ts": UNIX timestamp of the last unload (0 indicates none).
                    - "active_requests": Number of active requests (0).
            """
            return {
                "vram_loaded": True,
                "vram_last_load_ts": time.time() - 3600,
                "vram_last_unload_ts": 0,
                "active_requests": 0,
            }

        def get_handler(self, _model_id: str) -> None:
            """
            Retrieve a handler for the given model ID; this implementation always indicates no handler is available.

            This registry does not provide per-model handlers and therefore returns None for any model ID.
            """
            return

        async def request_vram_unload(self, _model_id: str) -> None:
            """
            Signal that a VRAM unload was requested for the given model by setting the internal event.

            Parameters:
                _model_id (str): Identifier of the model for which unload was requested. This parameter is ignored.
            """
            unloaded.set()

    registry = FakeRegistry()
    controller = CentralIdleAutoUnloadController(registry)
    try:
        controller.start()
        # Expect unload to be requested within a short timeout
        await asyncio.wait_for(unloaded.wait(), timeout=2)
    finally:
        await controller.stop()


@pytest.mark.asyncio
async def test_notify_activity_prevents_unload_with_fake_registry() -> None:
    """
    Verifies that notify_activity prevents the controller from requesting an unload when a handler reports recent activity.

    Sets up a fake registry and handler where the handler can report recent activity, starts the CentralIdleAutoUnloadController, signals recent activity via controller.notify_activity, and asserts that no unload request is made within a short window.
    """
    unloaded = asyncio.Event()

    class FakeHandler:
        def __init__(self) -> None:
            """
            Initialize the instance and set the default timeout.

            Sets the internal attribute `self._seconds` to 120.0, representing the default timeout value in seconds.
            """
            self._seconds = 120.0

        def seconds_since_last_activity(self) -> float:
            """
            Report how many seconds have elapsed since the last recorded activity.

            Returns:
                seconds (float): Number of seconds since the last recorded activity.
            """
            return self._seconds

        def set_recent(self) -> None:
            """
            Mark the handler as having recent activity by resetting its idle timer.

            This sets the internal seconds-since-last-activity counter to 0.0 to indicate immediate recent activity.
            """
            self._seconds = 0.0

    handler = FakeHandler()

    class FakeRegistry:
        def list_models(self) -> list[dict]:
            """
            Return the list of registered models.

            Each item is a dict with keys:
            - "id": model identifier string.
            - "metadata": dict containing model metadata; includes "auto_unload_minutes" (int) specifying idle minutes before auto-unload.

            Returns:
                list[dict]: List of model descriptors as described above.
            """
            return [{"id": "m1", "metadata": {"auto_unload_minutes": 1}}]

        def get_vram_status(self, _model_id: str) -> dict:
            """
            Provide a simulated VRAM status for the given model.

            Parameters:
                _model_id (str): Model identifier (ignored by this fake implementation).

            Returns:
                dict: A mapping with keys:
                    - "vram_loaded": `True` indicating VRAM is currently loaded.
                    - "vram_last_load_ts": UNIX timestamp of the last load (set to one hour ago).
                    - "vram_last_unload_ts": UNIX timestamp of the last unload (0 indicates none).
                    - "active_requests": Number of active requests (0).
            """
            return {
                "vram_loaded": True,
                "vram_last_load_ts": time.time() - 3600,
                "vram_last_unload_ts": 0,
                "active_requests": 0,
            }

        def get_handler(self, _model_id: str) -> FakeHandler:
            """
            Get the fake handler associated with the given model id.

            Parameters:
                _model_id (str): Model identifier used to look up the handler.

            Returns:
                FakeHandler: The fake handler instance associated with the model id.
            """
            return handler

        async def request_vram_unload(self, _model_id: str) -> None:
            """
            Signal that a VRAM unload was requested for the given model by setting the internal event.

            Parameters:
                _model_id (str): Identifier of the model for which unload was requested. This parameter is ignored.
            """
            unloaded.set()

    registry = FakeRegistry()
    controller = CentralIdleAutoUnloadController(registry)
    try:
        controller.start()

        # Simulate recent activity before controller attempts unload
        handler.set_recent()
        controller.notify_activity("m1")

        # Give the controller a short window to act; it should NOT call unload
        try:
            await asyncio.wait_for(unloaded.wait(), timeout=1)
            pytest.fail("Unload was triggered despite recent activity")
        except TimeoutError:
            # Expected: no unload call
            pass
    finally:
        await controller.stop()
