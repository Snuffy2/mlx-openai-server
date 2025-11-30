"""Model registry for managing multiple model handlers.

This module provides an asyncio-safe registry that tracks registered
models, attached per-model manager/handler objects, and VRAM-related
metadata. It exposes methods to attach managers via a loader callable,
request idempotent VRAM load/unload, and a per-request context manager
that increments/decrements an active request counter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import time
from typing import Any, TypedDict, cast

from loguru import logger

from ..schemas.model import ModelMetadata
from .manager_protocol import ManagerProtocol

_UNSET = object()


class VRAMStatus(TypedDict, total=False):
    """Per-model VRAM status metadata returned by ``get_vram_status``.

    Attributes
    ----------
    vram_loaded : bool
        Whether the model is currently loaded in VRAM.
    vram_last_load_ts : int or None
        Unix timestamp when the model was last loaded into VRAM, or ``None``.
    vram_last_unload_ts : int or None
        Unix timestamp when the model was last unloaded from VRAM, or ``None``.
    vram_last_request_ts : int or None
        Unix timestamp when the model last served a request, or ``None``.
    vram_load_error : str or None
        Error message from the last failed VRAM load attempt, or ``None``.
    active_requests : int
        Number of currently active requests being served by this model.
    -----
    The registry stores other dynamic keys as needed (e.g. ``_loading_task``),
    so this TypedDict is intentionally non-total to document the common subset
    surfaced to admin/UI code.
    """

    vram_loaded: bool
    vram_last_load_ts: int | None
    vram_last_unload_ts: int | None
    vram_last_request_ts: int | None
    vram_load_error: str | None
    active_requests: int


class ModelRegistry:
    """Asyncio event-loop-safe registry for model managers and metadata.

    The registry stores three parallel structures:
    - ``_handlers``: model_id -> manager object (or ``None`` if not
      attached yet)
    - ``_metadata``: model_id -> ``ModelMetadata`` instance
    - ``_extra``: model_id -> dict with VRAM and runtime metadata
    """

    def __init__(self) -> None:
        """
        Initialize an asyncio-safe registry for model managers, metadata, and per-model VRAM/runtime state.
        
        Sets up:
        - a mapping of model IDs to attached manager instances (or None),
        - stored ModelMetadata for each model,
        - a dynamic per-model extras dictionary for VRAM/runtime keys (common keys documented by `VRAMStatus`, but arbitrary runtime keys such as `_loading_task` may be present),
        - an asyncio.Lock for synchronizing registry mutations,
        - an optional synchronous activity notifier callable that receives a `model_id`.
        
        No return value.
        """
        self._handlers: dict[str, ManagerProtocol | None] = {}
        self._metadata: dict[str, ModelMetadata] = {}
        # Per-model VRAM/runtime metadata stored for admin/UI surfaces.
        # This is a dynamic mapping; use `VramMetadata` for documentation of
        # the common keys but allow other runtime keys (e.g. ``_loading_task``).
        self._extra: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        # Optional notifier for activity changes. Callable receives model_id.
        self._activity_notifier: Callable[[str], None] | None = None
        logger.info("Model registry initialized")

    def register_activity_notifier(self, notifier: Callable[[str], None]) -> None:
        """
        Register a synchronous notifier invoked when a model's active request count changes.
        
        Parameters:
            notifier (Callable[[str], None]): A lightweight callable that will be called with the `model_id`
                string whenever the active request count for that model changes.
        """
        self._activity_notifier = notifier

    def register_model(
        self,
        model_id: str,
        handler: ManagerProtocol | None,
        model_type: str,
        context_length: int | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a model in the registry and initialize its metadata.
        
        Registers a model identifier with an optional pre-attached manager, creates
        ModelMetadata (including creation timestamp), and initializes VRAM- and runtime-
        related metadata under the registry's internal stores.
        
        Parameters:
            model_id: Unique identifier for the model used by the registry and external endpoints.
            handler: Optional manager/handler instance to attach immediately, or `None` if not attached.
            model_type: Human-readable model type string (used in exposed metadata).
            context_length: Optional context length to include in the model metadata.
            metadata_extras: Optional mapping of additional metadata keys to merge into the model's metadata.
        
        Raises:
            ValueError: If `model_id` is already registered.
        """
        if model_id in self._handlers:
            raise ValueError(f"Model '{model_id}' is already registered")

        metadata = ModelMetadata(
            id=model_id,
            type=model_type,
            context_length=context_length,
            created_at=int(time.time()),
        )

        base_metadata: dict[str, Any] = {
            "model_type": model_type,
            "context_length": context_length,
            "status": "initialized" if handler else "unloaded",
        }
        if metadata_extras:
            base_metadata.update(metadata_extras)

        base_metadata.setdefault(
            "vram_loaded",
            bool(handler and getattr(handler, "is_vram_loaded", lambda: False)()),
        )
        base_metadata.setdefault("vram_last_load_ts", None)
        base_metadata.setdefault("vram_last_unload_ts", None)
        base_metadata.setdefault("vram_last_request_ts", None)
        base_metadata.setdefault("vram_load_error", None)
        base_metadata.setdefault("active_requests", 0)

        self._handlers[model_id] = handler
        self._metadata[model_id] = metadata
        self._extra[model_id] = base_metadata

    async def update_model_state(
        self,
        model_id: str,
        *,
        handler: ManagerProtocol | None | object = _UNSET,
        status: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """
        Update the attached manager and merge extra metadata for a registered model.
        
        Parameters:
            model_id (str): Registered model identifier.
            handler (ManagerProtocol | None | object): Manager to attach, `None` to detach, or the sentinel `_UNSET` to leave the handler unchanged.
            status (str | None): Optional status string to set in the model's extra metadata.
            metadata_updates (dict[str, Any] | None): Optional mapping of extra metadata to merge into the model's runtime extras.
        
        Raises:
            KeyError: If `model_id` is not registered in the registry.
        """
        async with self._lock:
            if model_id not in self._metadata:
                raise KeyError(f"Model '{model_id}' not found in registry")

            entry = self._extra.setdefault(model_id, {})

            if handler is not _UNSET:
                # Handler may be ``object`` when the sentinel _UNSET is allowed;
                # cast to the manager protocol type for the registry storage.
                self._handlers[model_id] = cast("ManagerProtocol | None", handler)
                if handler is not None:
                    # refresh created_at to indicate new attachment
                    self._metadata[model_id].created_at = int(time.time())
                    # Update vram_loaded status
                    try:
                        loaded = bool(getattr(handler, "is_vram_loaded", lambda: False)())
                    except Exception:
                        loaded = False
                    entry["vram_loaded"] = loaded
                    if loaded:
                        entry["vram_last_load_ts"] = int(time.time())
                    else:
                        entry.pop("vram_last_load_ts", None)
                    entry["status"] = "loaded" if loaded else "unloaded"
                else:
                    # Handler detached
                    entry["vram_loaded"] = False
                    entry.pop("vram_last_load_ts", None)

            if metadata_updates:
                entry.update(metadata_updates)
            if status is not None:
                entry["status"] = status

    async def unregister_model(self, model_id: str) -> None:
        """
        Unregisters a model and removes all associated metadata and runtime state from the registry.
        
        Parameters:
            model_id (str): Identifier of the model to remove.
        
        Raises:
            KeyError: If the model_id is not registered.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            del self._handlers[model_id]
            del self._metadata[model_id]
            self._extra.pop(model_id, None)
            logger.info(f"Unregistered model: {model_id}")

    def get_handler(self, model_id: str) -> ManagerProtocol | None:
        """
        Retrieve the manager attached to the given model ID.
        
        Returns:
            The attached `ManagerProtocol` instance, or `None` if no manager is attached.
        
        Raises:
            KeyError: If `model_id` is not registered in the registry.
        """
        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._handlers[model_id]

    def list_models(self) -> list[dict[str, Any]]:
        """
        List registered models as OpenAI-compatible metadata dictionaries.
        
        Each dictionary contains the keys "id", "object", "created", and "owned_by". If extra metadata exists for a model, it is included under the optional "metadata" key.
        
        Returns:
            list[dict[str, Any]]: A list of model metadata dictionaries.
        """
        output: list[dict[str, Any]] = []
        for mid, metadata in self._metadata.items():
            entry: dict[str, Any] = {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
            }
            extra = self._extra.get(mid)
            if extra:
                entry["metadata"] = extra
            output.append(entry)
        return output

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """
        Retrieve the stored ModelMetadata for a registered model.
        
        Parameters:
            model_id (str): The identifier of the model.
        
        Returns:
            ModelMetadata: The metadata associated with `model_id`.
        
        Raises:
            KeyError: If `model_id` is not registered in the registry.
        """
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    async def get_or_attach_manager(
        self,
        model_id: str,
        loader: Callable[[str], Awaitable[ManagerProtocol]],
        *,
        timeout: float | None = None,
    ) -> ManagerProtocol:
        """
        Get the manager for the given model, attaching and awaiting the provided loader if no manager is currently attached.
        
        Parameters:
            model_id (str): Registered model identifier.
            loader (Callable[[str], Awaitable[ManagerProtocol]]): Async callable that accepts `model_id` and returns a manager instance; used when no manager is attached.
            timeout (float | None): Optional timeout in seconds to wait for the loader task.
        
        Returns:
            ManagerProtocol: The attached or newly created manager instance.
        
        Raises:
            KeyError: If `model_id` is not registered.
            Exception: Any exception raised by the loader is propagated after recording the load error in the model's metadata.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            manager = self._handlers[model_id]
            if manager is not None:
                return manager

            loading = self._extra.setdefault(model_id, {}).get("_loading_task")
            if loading is None:
                # Use ensure_future to accept any Awaitable (not only coroutines)
                task = asyncio.ensure_future(loader(model_id))
                self._extra[model_id]["_loading_task"] = task
            else:
                task = loading

        try:
            if timeout is not None:
                manager = await asyncio.wait_for(task, timeout=timeout)
            else:
                manager = await task
        except Exception as exc:
            async with self._lock:
                entry = self._extra.setdefault(model_id, {})
                entry["vram_load_error"] = str(exc)
                entry.pop("_loading_task", None)
            raise

        async with self._lock:
            self._handlers[model_id] = manager
            entry = self._extra.setdefault(model_id, {})
            entry.setdefault("active_requests", 0)
            try:
                loaded = bool(getattr(manager, "is_vram_loaded", lambda: False)())
            except Exception:
                loaded = False
            entry["vram_loaded"] = loaded
            if loaded:
                entry["vram_last_load_ts"] = int(time.time())
            entry.pop("_loading_task", None)

        return manager

    async def request_vram_load(
        self,
        model_id: str,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:
        """
        Request the attached manager to load the model's weights into VRAM.
        
        Calls the model's manager to ensure weights are resident in VRAM and updates the registry's VRAM metadata on success.
        
        Parameters:
            model_id (str): Registered model identifier.
            force (bool): If True, force a reload even when the model is already marked loaded.
            timeout (float | None): Maximum seconds to wait for the manager operation; no timeout if None.
        
        Raises:
            KeyError: If the model is not registered or no manager is attached.
            RuntimeError: If the manager's load operation fails.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")
            manager = self._handlers[model_id]
            entry = self._extra.setdefault(model_id, {})

        if manager is None:
            raise KeyError(f"No manager attached for model '{model_id}'")

        coro = manager.ensure_vram_loaded(force=force)
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

        async with self._lock:
            entry = self._extra.setdefault(model_id, {})
            entry["vram_loaded"] = True
            entry["vram_last_load_ts"] = int(time.time())
            # Keep human-readable status in sync with VRAM residency
            entry["status"] = "loaded"
            entry.pop("vram_load_error", None)

    async def request_vram_unload(self, model_id: str, *, timeout: float | None = None) -> None:
        """
        Request that the attached manager release the model's VRAM resources.
        
        Parameters:
            model_id (str): Registered model identifier.
            timeout (float | None): Optional timeout in seconds to wait for the manager operation.
        
        Raises:
            KeyError: If the model is not registered or no manager is attached.
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")
            manager = self._handlers[model_id]
            entry = self._extra.setdefault(model_id, {})

        if manager is None:
            raise KeyError(f"No manager attached for model '{model_id}'")

        coro = manager.release_vram()
        if timeout is not None:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro

        async with self._lock:
            entry = self._extra.setdefault(model_id, {})
            entry["vram_loaded"] = False
            entry["vram_last_unload_ts"] = int(time.time())
            # Keep human-readable status in sync with VRAM residency
            entry["status"] = "unloaded"
            entry.pop("vram_load_error", None)

    def handler_session(
        self,
        model_id: str,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[ManagerProtocol]:
        """
        Provide an async context manager that yields the model's attached manager for the duration of a single request session.
        
        Parameters:
            model_id (str): Registered model identifier.
            ensure_vram (bool): If True, ensure the manager has loaded model weights into VRAM before yielding.
            ensure_timeout (float | None): Optional timeout, in seconds, applied to the manager's VRAM ensure call.
        
        Yields:
            ManagerProtocol: The attached manager instance for the duration of the context.
        
        Returns:
            An async context manager that yields the manager for a request session.
        
        Notes:
            The registry increments the per-model `active_requests` counter on entry and decrements it on exit. When `active_requests` drops to zero the configured activity notifier (if any) is invoked.
        """

        @asynccontextmanager
        async def _session() -> AsyncIterator[Any]:
            """
            Create an async context for a model handler session that tracks active requests, optionally ensures model VRAM is loaded, and yields the attached manager.
            
            On entry, increments the model's active request count and updates the last-request timestamp; calls the registered activity notifier (if any). If the model is not registered, raises KeyError. If no manager is attached, decrements the active request count and raises KeyError. If VRAM should be ensured, waits for the manager to load weights before yielding. On exit, decrements the active request count and, if it drops to zero, calls the activity notifier. Exceptions raised by the notifier are caught and logged.
            
            Returns:
                The attached manager for the requested model (yielded to the caller).
            """
            async with self._lock:
                if model_id not in self._handlers:
                    raise KeyError(f"Model '{model_id}' not found in registry")
                manager = self._handlers[model_id]
                entry = self._extra.setdefault(model_id, {})
                entry["active_requests"] = entry.get("active_requests", 0) + 1
                entry["vram_last_request_ts"] = int(time.time())

            # Notify activity (reset idle timers) for this model.
            try:
                if self._activity_notifier:
                    self._activity_notifier(model_id)
            except Exception:
                # Notifier should never raise; log and continue.
                logger.exception("Activity notifier raised an exception")

            if manager is None:
                async with self._lock:
                    entry["active_requests"] = max(0, entry.get("active_requests", 1) - 1)
                raise KeyError(f"No manager attached for model '{model_id}'")

            if ensure_vram:
                coro = manager.ensure_vram_loaded()
                if ensure_timeout is not None:
                    await asyncio.wait_for(coro, timeout=ensure_timeout)
                else:
                    await coro

            try:
                yield manager
            finally:
                async with self._lock:
                    entry = self._extra.setdefault(model_id, {})
                    entry["active_requests"] = max(0, entry.get("active_requests", 1) - 1)
                    # If active requests dropped to zero, notify controller so it
                    # can begin idle countdown for this model.
                    if entry.get("active_requests", 0) == 0:
                        try:
                            if self._activity_notifier:
                                self._activity_notifier(model_id)
                        except Exception:
                            logger.exception("Activity notifier raised an exception")

        return _session()

    def get_vram_status(self, model_id: str) -> dict[str, Any]:
        """
        Get VRAM-related status for the specified model.
        
        Returns a dictionary with the keys: `vram_loaded`, `vram_last_load_ts`, `vram_last_unload_ts`,
        `vram_last_request_ts`, `vram_load_error`, and `active_requests`.
        
        Parameters:
            model_id (str): Registered model identifier.
        
        Returns:
            dict[str, Any]: VRAM status fields for the model.
        
        Raises:
            KeyError: If `model_id` is not registered in the registry.
        """
        if model_id not in self._extra:
            raise KeyError(f"Model '{model_id}' not found in registry")
        entry = self._extra[model_id]
        return {
            "vram_loaded": bool(entry.get("vram_loaded", False)),
            "vram_last_load_ts": entry.get("vram_last_load_ts"),
            "vram_last_unload_ts": entry.get("vram_last_unload_ts"),
            "vram_last_request_ts": entry.get("vram_last_request_ts"),
            "vram_load_error": entry.get("vram_load_error"),
            "active_requests": int(entry.get("active_requests", 0)),
        }

    def has_model(self, model_id: str) -> bool:
        """
        Check whether a model identifier is registered in the registry.
        
        Returns:
            `True` if the given `model_id` is registered, `False` otherwise.
        """
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """
        Get the number of registered models.
        
        Returns:
            count (int): Number of models currently registered in the registry.
        """
        return len(self._handlers)