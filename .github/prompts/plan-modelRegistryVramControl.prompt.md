Plan: ModelRegistry VRAM Status & Manager-Controlled Load/Unload

Goal

Replace network proxy routing (hub worker HTTP forwarding) with direct in-process routing to handler/manager objects. Provide idempotent VRAM load/unload controls that:
- Let the status page and CLI show whether a model is loaded into VRAM.
- Allow admins to request load/unload through the registry/controller (no auth for now).
- Keep VRAM operations owned and executed by the LazyHandler/LazyHandlerManager.

Terminology

- start/stop: process-level lifecycle (hub daemon / worker processes).
- load/unload: VRAM binding/unbinding (JIT model weight residency), performed by manager objects.

Design Overview

- ModelRegistry remains the single source of truth for registered models and metadata. It will store manager objects (not raw handlers), expose per-model metadata (including VRAM state and active request counters), and provide admin helpers that call into manager methods to request VRAM operations.
- Managers (LazyHandlerManager / LazyHandler) remain the authority for VRAM load/unload. They must implement a small interface so the registry and admin endpoints can call them uniformly.
- IdleAutoUnloadController remains responsible for timing/deciding when to actually unload VRAM; registry will notify it when active requests drop to zero (or the controller can be passed into registry wiring).

Registry API (new public methods)

1) async def get_or_attach_manager(self, model_id: str, loader: Callable[[str], Awaitable[Any]], *, timeout: float | None = None) -> Any

- Return the manager object for `model_id`, attaching it if not present.
- Idempotent: if already attached return immediately.
- Raises: KeyError if model not registered; RuntimeError on loader failure.

2) async def request_vram_load(self, model_id: str, *, force: bool = False, timeout: float | None = None) -> None

- Ask the attached manager to ensure VRAM residency.
- Idempotent: no-op if already loaded and force=False.
- On success update `_extra[model_id]` with `vram_loaded=True` and `vram_last_load_ts`.
- On failure set `_extra[model_id]['vram_load_error']` and raise RuntimeError.

3) async def request_vram_unload(self, model_id: str, *, timeout: float | None = None) -> None

- Ask manager to release VRAM.
- Idempotent: no-op if already unloaded.
- On success set `vram_loaded=False` and `vram_last_unload_ts`.

4) async def handler_session(self, model_id: str, *, ensure_vram: bool = True, ensure_timeout: float | None = None) -> AsyncContextManager[Any]

- Async context manager for per-request lifetime; increments `active_requests` on enter and decrements on exit.
- If ensure_vram True, calls manager.ensure_vram_loaded() on enter (idempotent).
- Notifies IdleAutoUnloadController when `active_requests` becomes zero.

5) async def get_vram_status(self, model_id: str) -> dict[str, Any]

- Returns `vram_loaded`, `vram_last_load_ts`, `vram_last_unload_ts`, `vram_load_error`, `active_requests`.

Registry metadata keys to add (`_extra[model_id]`)

- vram_loaded : bool
- vram_last_load_ts : int | None
- vram_last_unload_ts : int | None
- vram_load_error : str | None
- active_requests : int

Manager (LazyHandler) Interface

Managers must implement the following methods so the registry and admin endpoints can call them safely. Managers remain responsible for concurrency and internal JIT semantics.

- def is_vram_loaded(self) -> bool
- async def ensure_vram_loaded(self, *, force: bool = False, timeout: float | None = None) -> None
- async def release_vram(self, *, timeout: float | None = None) -> None
- async def request_session(self, *, ensure_vram: bool = True, ensure_timeout: float | None = None) -> AsyncContextManager[Any]

Behavioral Contracts

- All load/unload operations MUST be idempotent.
- ensure_vram_loaded must be safe for concurrent calls and should not double-initialize memory allocations.
- request_session should manage a per-manager active request counter and optionally start an idle timer when the counter hits zero; registry will mirror the counter in `_extra` for admin/UI visibility.
- Registry must not itself perform VRAM operations; it only calls the manager's methods and records state.

Admin Endpoints & CLI (no auth for now)

- POST /hub/models/{model_id}/vram/load  -> calls registry.request_vram_load
- POST /hub/models/{model_id}/vram/unload -> calls registry.request_vram_unload
- GET /hub/models -> include VRAM state in returned model metadata

Return semantics

- For fast operations return 200 on success.
- For potentially long-running operations, return 202 Accepted and execute the manager operation asynchronously (optionally add a status polling endpoint). For now prefer synchronous with a sensible timeout (e.g., 60s) and return 503/504 on timeout/error mapped via `create_error_response`.

Integration with IdleAutoUnloadController

- Registry notifies controller when `active_requests` becomes zero (or controller subscribes to registry updates). Controller decides when to call `registry.request_vram_unload` (or instruct managers directly) to perform idle unload.
- Controller remains responsible for timing, heuristics and preventing thrashing.

Testing

- Add unit tests to `tests/test_model_registry.py` to cover get_or_attach_manager (with mock loader), request_vram_load/unload idempotency, and handler_session context behavior.
- Update/replace `tests/test_hub_proxy.py` to reflect direct routing (mock registry and manager objects rather than spinning worker HTTP endpoints).
- Add tests for admin endpoints in `tests/test_hub_service_api.py` to assert VRAM status exposure.

Migration Notes

- Keep `app/hub/proxy.py` available behind a feature flag or conditionally mounted until clients/tests are migrated.
- Ensure that `app.server` wiring injects `app.state.model_registry` and `app.state.hub_controller` so endpoints can pass the controller's manager-creation function as the loader.

Example usage patterns

- Endpoint on request:

  manager = await registry.get_or_attach_manager(model_id, loader=app.state.hub_controller.create_manager)
  async with registry.handler_session(model_id) as manager:
      return await manager.handle_chat_stream(request)

- Admin-triggered load:

  await registry.request_vram_load(model_id, force=False, timeout=60.0)

Next steps

- Add typed stubs to `app/core/model_registry.py` and manager interface stubs under `app/handler/`.
- Update endpoints and tests incrementally, keeping backward compatibility until CI passes.

