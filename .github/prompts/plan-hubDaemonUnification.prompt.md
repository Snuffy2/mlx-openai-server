## Plan: Single-Port Hub Daemon

Unify hub mode so a single FastAPI daemon (with existing handler pipeline) listens on `DEFAULT_PORT`, serves `/v1/*`, `/hub/*`, and the HTML dashboard, while background workers keep their own `DEFAULT_MODEL_STARTING_PORT` assignments. Remove the unused proxy layer plus `daemon_port` plumbing, ensure CLI `launch` semantics remain untouched, and make hub startup fail loudly when `DEFAULT_PORT` is already occupied.

### Steps
1. Remove `app/hub/proxy.py` and its router wiring from `app/server.py`, deleting proxy-focused tests (`tests/test_hub_proxy.py`) and any docs referencing the intermediary gateway.
2. Strip `daemon_port` from `app/hub/config.py`, `app/cli.py`, runtime JSON, and related tests (`tests/test_hub_runtime.py`, CLI hub suites), ensuring hub CLI commands now record only PID/host while workers still allocate from `DEFAULT_MODEL_STARTING_PORT`.
3. Refactor `app/hub/daemon.py` so it reuses the OpenAI API routing/middleware stack (e.g., by calling `configure_fastapi_app`) and exposes `/v1/*`, `/hub/*`, and `/hub` HTML directly; confirm status endpoints and handler actions operate through the existing `ModelRegistry`/manager hooks.
4. Keep `launch` command behavior untouched by isolating changes to hub-specific paths; once daemon serves everything, delete redundant `setup_server` if truly unused, otherwise leave the single-model path intact.
5. Add an explicit bind check so daemon startup errors when `DEFAULT_PORT` is in use, document the new single-port behavior in `docs/HUB_MODE.md` and `README.md`, and update templates/status text if messaging changed.

### Further Considerations
1. Confirm no other modules depend on proxy semantics (search for `hub_daemon_url`, `/v1` interception) before removal.
2. Decide whether to gate `setup_server` removal behind tests proving `launch` path parity, or keep it as a non-hub helper for now.
