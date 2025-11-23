# Hub Roadmap Tracker (temporary)

_Last updated: 2025-11-22_

Keep this file in sync whenever progress is made so work can resume even if the chat session is lost.

## Remaining Steps

1. **Docs/tests for orchestration features**  
   - Goal: update the README and reference docs with the final hub.yaml schema, log-path defaults, reload semantics, CLI usage, `/hub` behavior, and flash messaging, plus add regression tests covering config parsing, slug validation, log defaults, YAML reload logic, CLI subcommands, `/hub` POST idempotence, group-capacity 429s, exit-code health checks, and log-rotation adherence.  
   - Status: Not started.

## Completed Steps

- **2025-11-22** – Delivered `/hub` action endpoints plus flash messaging: FastAPI now exposes service-backed POST routes for start/stop/reload/load/unload, reloads hub.yaml on each request for accurate counts, surfaces flash/toast messaging in the dashboard, and wires the HTML controls to those endpoints. Added regression coverage via `tests/test_hub_service_api.py` and tightened hub config parsing to emit correct 429s.
- **2025-11-22** – Added hub integration tests that exercise FastAPI routes with multiple stubbed models, covering happy path routing, missing `model` validation, and controller error surfacing.
- **2025-11-22** – Implemented hub CLI load/unload/watch commands plus the backing FastAPI endpoints, README/docs updates, and regression tests.
- **2025-11-22** – Delivered the `/hub` HTML status page, including the FastAPI route, dashboard UI, documentation updates, and regression tests guarding the new surface.
- **2025-11-22** – Shipped the process-based hub manager (`app/hub/manager.py`) with YAML reload diffing, per-group capacity enforcement, crash detection, log redirection, and regression tests in `tests/test_hub_manager.py`.
- **2025-11-22** – Completed per-model observability polish: introduced `HubObservabilitySink`, contextual logging bindings, per-model log directories, documentation (`docs/HUB_OBSERVABILITY.md`), and regression tests guarding the sink.
- **2025-11-22** – Enabled interactive `/hub` dashboard controls with load/unload buttons, toast feedback, README updates, and client-side wiring to the existing model action endpoints.
- **2025-11-22** – Improved `hub watch` output with tabular columns, uptime/readability metrics, and clearer error guidance; authentication requirements dropped per updated plan.
- **2025-11-22** – Finished CLI orchestration polish: hub commands now emit flash-style notices, call `reload` before start/stop/load/unload/status, and gained regression tests covering the new behavior.

## Notes
- Update this file after completing any milestone or discovering new work.
- Remove items once they are fully delivered and verified.
