"""Manager protocol used across the core modules.

This module defines a lightweight typing contract for manager-like objects
that control VRAM residency and per-request sessions. Placing it under
`app.core` keeps the handler package focused on concrete MLX handler
implementations.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol


class ManagerProtocol(Protocol):
    """Protocol describing manager behaviour used by the ModelRegistry.

    Implementations must ensure their VRAM operations are safe to call
    concurrently and idempotent where appropriate.
    """

    def is_vram_loaded(self) -> bool:  # pragma: no cover - typing stub
        """
        Report whether the manager currently has model weights resident in VRAM.
        
        Returns:
            bool: `True` if the manager has model weights loaded in VRAM, `False` otherwise.
        """

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:  # pragma: no cover - typing stub
        """
        Ensure the manager's model weights are resident in VRAM.
        
        This operation should be safe to call multiple times (idempotent) and implementations are expected to handle concurrent callers appropriately.
        
        Parameters:
        	force (bool): If True, force reloading or re-establishing VRAM residency even if already loaded.
        	timeout (float | None): Maximum time in seconds to wait for VRAM residency; None means no timeout.
        """

    async def release_vram(
        self,
        *,
        timeout: float | None = None,
    ) -> None:  # pragma: no cover - typing stub
        """
        Release VRAM resources held by this manager.
        
        This operation frees any GPU/VRAM residency associated with the manager. Implementations should be safe to call when no resources are currently resident (i.e., act as a no-op) and should be idempotent where possible.
        
        Parameters:
            timeout (float | None): Maximum number of seconds to wait for the release to complete. If `None`, wait without a timeout.
        """

    def request_session(
        self,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[Any]:  # pragma: no cover - typing stub
        """
        Provide an async context manager for a per-request session that manages active-request counters and optionally ensures VRAM residency.
        
        Parameters:
            ensure_vram (bool): If True, ensure the manager's model weights are resident in VRAM before yielding the session.
            ensure_timeout (float | None): Optional timeout, in seconds, applied to the VRAM ensure operation.
        
        Returns:
            AbstractAsyncContextManager[Any]: An async context manager that increments an internal active-request counter on enter and decrements it on exit; when `ensure_vram` is True it ensures VRAM residency before yielding.
        """