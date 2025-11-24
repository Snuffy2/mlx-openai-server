"""Hub daemon scaffold: FastAPI app factory and HubSupervisor skeleton.

This module provides a non-complete but useful scaffold for the hub daemon
supervisor and HTTP control API. Implementations that require deeper
integration with model handlers should expand the supervisor methods; tests
may mock the supervisor where appropriate.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from loguru import logger

from .config import MLXHubConfig, load_hub_config


@dataclass
class ModelRecord:
    """Runtime record for a supervised model.

    Attributes
    ----------
    name : str
        Slug name of the model.
    config : Any
        The model configuration object (from `MLXHubConfig`).
    process : asyncio.subprocess.Process | None
        The subprocess instance if started.
    pid : int | None
        OS process id when running.
    port : int | None
        Assigned port for the model process.
    started_at : float | None
        Epoch timestamp when process was started.
    exit_code : int | None
        Last exit code, if the process exited.
    memory_loaded : bool
        Whether the model's runtime/memory is currently loaded.
    auto_unload_minutes : int | None
        Optional idle minutes after which memory should be auto-unloaded.
    group : str | None
        Group slug for capacity accounting.
    is_default : bool
        Whether this model is marked as a default auto-start.
    model_path : str | None
        Configured model path.
    """

    name: str
    config: Any
    process: asyncio.subprocess.Process | None = None
    pid: int | None = None
    port: int | None = None
    started_at: float | None = None
    exit_code: int | None = None
    memory_loaded: bool = False
    auto_unload_minutes: int | None = None
    group: str | None = None
    is_default: bool = False
    model_path: str | None = None


class HubSupervisor:
    """Supervise model worker processes and runtime state.

    This is a conservative scaffold that implements non-blocking process
    management patterns. Long-running operations should be scheduled as
    background tasks when invoked from FastAPI endpoints.
    """

    def __init__(self, hub_config: MLXHubConfig) -> None:
        self.hub_config = hub_config
        self._models: dict[str, ModelRecord] = {}
        self._lock = asyncio.Lock()
        self._bg_tasks: list[asyncio.Task] = []
        self._shutdown = False

        # Populate model records from hub_config (best-effort)
        for model in getattr(hub_config, "models", []):
            name = getattr(model, "name", None) or str(model)
            record = ModelRecord(
                name=name,
                config=model,
                port=getattr(model, "port", None),
                group=getattr(model, "group", None),
                is_default=getattr(model, "is_default_model", False),
                model_path=getattr(model, "model_path", None),
            )
            self._models[name] = record

    async def start_model(self, name: str) -> dict[str, Any]:
        """Start the model worker as an OS subprocess.

        The supervisor expects a model-specific start command to be available
        in the model config under `start_command` (a list of argv). If no
        command is present, a HTTPException is raised.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]

            if record.process is not None and record.process.returncode is None:
                return {"status": "already_running", "name": name, "pid": record.pid}

            cmd = None
            cfg = record.config
            # Expect a start_command attribute (list) on model config
            cmd = getattr(cfg, "start_command", None)
            if not cmd:
                raise HTTPException(status_code=400, detail="no start_command defined for model")

            # Spawn subprocess without blocking the event loop
            proc = await asyncio.create_subprocess_exec(*cmd)
            record.process = proc
            record.pid = proc.pid
            record.started_at = time.time()
            record.exit_code = None

            # schedule a watcher
            task = asyncio.create_task(self._watch_process(name, proc))
            self._bg_tasks.append(task)

            logger.info(f"Started model {name} pid={proc.pid}")
            return {"status": "started", "name": name, "pid": proc.pid}

    async def stop_model(self, name: str) -> dict[str, Any]:
        """Stop a supervised model process.

        Parameters
        ----------
        name : str
            Slug name of the model to stop.

        Returns
        -------
        dict[str, Any]
            A status dict describing the result (not_running/stopped).

        Raises
        ------
        HTTPException
            If the model is not found.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            proc = record.process
            if proc is None or proc.returncode is not None:
                return {"status": "not_running", "name": name}

            with contextlib.suppress(ProcessLookupError):
                proc.terminate()

        # Wait outside the lock
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

        record.exit_code = proc.returncode
        record.process = None
        record.pid = None
        logger.info(f"Stopped model {name} exit_code={record.exit_code}")
        return {"status": "stopped", "name": name, "exit_code": record.exit_code}

    async def load_model_memory(self, name: str, reason: str = "manual") -> dict[str, Any]:
        """Mark a model's memory/runtime as loaded.

        This is a lightweight marker; heavy loading work should be scheduled
        separately as background tasks integrated with model handlers.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            # Mark loaded and schedule any heavy lifting as a background task
            record.memory_loaded = True
            logger.info(f"Marked model {name} memory_loaded (reason={reason})")
            return {"status": "memory_loaded", "name": name}

    async def unload_model_memory(self, name: str, reason: str = "manual") -> dict[str, Any]:
        """Mark a model's memory/runtime as unloaded.

        The supervisor will not attempt to free in-process handler state here;
        this method provides a consistent API surface for the CLI and tests.
        """

        async with self._lock:
            if name not in self._models:
                raise HTTPException(status_code=404, detail="model not found")
            record = self._models[name]
            record.memory_loaded = False
            logger.info(f"Marked model {name} memory_unloaded (reason={reason})")
            return {"status": "memory_unloaded", "name": name}

    async def reload_config(self) -> dict[str, Any]:
        """Reload the hub configuration and reconcile models.

        This implementation performs a best-effort reload: it replaces the
        in-memory hub_config and returns a simple diff.
        """

        new_hub = load_hub_config(self.hub_config.source_path)
        old_names = set(self._models.keys())
        new_names = {getattr(m, "name", str(m)) for m in getattr(new_hub, "models", [])}

        started = list(new_names - old_names)
        stopped = list(old_names - new_names)
        unchanged = list(old_names & new_names)

        # Rebuild records for new config (conservative)
        self.hub_config = new_hub
        self._models = {}
        for model in getattr(new_hub, "models", []):
            name = getattr(model, "name", None) or str(model)
            record = ModelRecord(name=name, config=model, port=getattr(model, "port", None))
            self._models[name] = record

        logger.info(f"Reloaded hub config: started={started} stopped={stopped}")
        return {"started": started, "stopped": stopped, "unchanged": unchanged}

    def get_status(self) -> dict[str, Any]:
        """Return a serializable snapshot of supervisor state.

        The returned dict includes a `timestamp` and a `models` list where
        each model object contains keys used by the CLI and status UI.
        """

        snapshot = {
            "timestamp": time.time(),
            "models": [],
        }
        for name, rec in self._models.items():
            state = "running" if rec.process and rec.process.returncode is None else "stopped"
            snapshot["models"].append(
                {
                    "name": name,
                    "state": state,
                    "pid": rec.pid,
                    "port": rec.port,
                    "started_at": rec.started_at,
                    "exit_code": rec.exit_code,
                    "memory_loaded": rec.memory_loaded,
                    "group": rec.group,
                    "is_default_model": rec.is_default,
                    "model_path": rec.model_path,
                    "auto_unload_minutes": rec.auto_unload_minutes,
                }
            )
        return snapshot

    async def _watch_process(self, name: str, proc: asyncio.subprocess.Process) -> None:
        try:
            await proc.wait()
            async with self._lock:
                rec = self._models.get(name)
                if rec is not None:
                    rec.exit_code = proc.returncode
                    rec.process = None
                    rec.pid = None
                    logger.info(f"Model process exited: {name} code={proc.returncode}")
        except asyncio.CancelledError:
            logger.debug(f"Watcher cancelled for {name}")

    async def shutdown_all(self) -> None:
        """Gracefully stop all supervised model processes.

        This performs a best-effort shutdown of each supervised process and
        logs failures without raising to the caller.
        """

        logger.info("Shutting down all supervised model processes")
        async with self._lock:
            names = list(self._models.keys())
        for name in names:
            try:
                await self.stop_model(name)
            except Exception as exc:  # pragma: no cover - best-effort shutdown
                logger.exception(f"Error stopping model {name}: {exc}")


def create_app(hub_config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application for the hub daemon.

    Parameters
    ----------
    hub_config_path : str | None
        Optional path to the hub YAML configuration. When None, the default
        path from `app/hub/config.py` is used.

    Returns
    -------
    FastAPI
        Configured FastAPI app instance with supervisor attached at
        `app.state.supervisor`.
    """

    app = FastAPI(title="mlx hub daemon")

    hub_config = load_hub_config(hub_config_path)
    supervisor = HubSupervisor(hub_config)
    app.state.supervisor = supervisor

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Hub daemon starting up")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Hub daemon shutting down")
        await supervisor.shutdown_all()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/hub/status")
    async def hub_status() -> dict[str, Any]:
        return supervisor.get_status()

    @app.post("/hub/reload")
    async def hub_reload() -> dict[str, Any]:
        return await supervisor.reload_config()

    @app.post("/hub/shutdown")
    async def hub_shutdown(background_tasks: BackgroundTasks) -> dict[str, str]:
        background_tasks.add_task(supervisor.shutdown_all)
        return {"status": "shutdown_scheduled"}

    @app.post("/hub/models/{name}/start")
    async def model_start(name: str) -> dict[str, Any]:
        return await supervisor.start_model(name)

    @app.post("/hub/models/{name}/stop")
    async def model_stop(name: str) -> dict[str, Any]:
        return await supervisor.stop_model(name)

    @app.post("/hub/models/{name}/load-model")
    async def model_load(name: str, request: Request) -> dict[str, Any]:
        payload = (
            await request.json()
            if request.headers.get("content-type") == "application/json"
            else {}
        )
        reason = payload.get("reason", "cli") if isinstance(payload, dict) else "cli"
        return await supervisor.load_model_memory(name, reason)

    @app.post("/hub/models/{name}/unload-model")
    async def model_unload(name: str, request: Request) -> dict[str, Any]:
        payload = (
            await request.json()
            if request.headers.get("content-type") == "application/json"
            else {}
        )
        reason = payload.get("reason", "cli") if isinstance(payload, dict) else "cli"
        return await supervisor.unload_model_memory(name, reason)

    return app
