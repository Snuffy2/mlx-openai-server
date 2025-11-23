"""FastAPI/uvicorn harness for hub-managed deployments."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
import sys

from fastapi import FastAPI
from loguru import logger
import uvicorn

from ..core.model_registry import ModelRegistry
from ..server import configure_fastapi_app, configure_logging
from ..version import __version__
from .config import MLXHubConfig
from .controller import HubController
from .runtime import HubRuntime


def setup_hub_server(runtime: HubRuntime) -> uvicorn.Config:
    """Create the FastAPI application used by hub deployments.

    Parameters
    ----------
    runtime : HubRuntime
        Runtime object encapsulating hub configuration and state trackers.

    Returns
    -------
    uvicorn.Config
        Configured uvicorn configuration ready to be passed to ``uvicorn.Server``.
    """

    log_file = runtime.config.log_path / "hub.log"
    configure_logging(log_file=str(log_file), no_log_file=False, log_level=runtime.config.log_level)

    registry = ModelRegistry()
    controller = HubController(runtime, registry)
    app = FastAPI(
        title="MLX Hub API",
        description="Multi-model hub for the MLX OpenAI-compatible server",
        version=__version__,
        lifespan=_create_hub_lifespan(controller),
    )

    app.state.registry = registry
    app.state.server_config = runtime.config
    app.state.hub_controller = controller
    app.state.hub_runtime = runtime
    app.state.model_metadata = []

    configure_fastapi_app(app)

    logger.info(
        "Starting hub server on %s:%s with %d configured model(s)",
        runtime.config.host,
        runtime.config.port,
        len(runtime.model_names()),
    )

    return uvicorn.Config(
        app=app,
        host=runtime.config.host,
        port=runtime.config.port,
        log_level=runtime.config.log_level.lower(),
        access_log=True,
    )


def _create_hub_lifespan(
    controller: HubController,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Return a FastAPI lifespan context that owns the hub controller.

    Parameters
    ----------
    controller : HubController
        Controller responsible for coordinating handler managers.

    Returns
    -------
    Callable[[FastAPI], AbstractAsyncContextManager[None]]
        ``FastAPI`` lifespan hook that starts/stops the controller.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            await controller.start()
            yield
        finally:
            await controller.shutdown()

    return lifespan


def print_hub_startup_banner(config: MLXHubConfig) -> None:
    """Emit a concise startup banner for hub deployments.

    Parameters
    ----------
    config : MLXHubConfig
        Hub configuration describing host/port/logging defaults.
    """

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✨ MLX Hub Server v%s Starting ✨", __version__)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🌐 Host: %s", config.host)
    logger.info("🔌 Port: %s", config.port)
    logger.info("📝 Log Level: %s", config.log_level)
    logger.info("📁 Log Path: %s", config.log_path)
    logger.info("📦 Models: %d configured", len(config.models))
    defaults: list[str] = []
    for model in config.models:
        if not model.is_default_model:
            continue
        name = model.name
        if name:
            defaults.append(name)
    if defaults:
        logger.info("⭐ Default Models: %s", ", ".join(defaults))
    else:
        logger.info("⭐ Default Models: none (all on-demand)")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


async def start_hub(runtime: HubRuntime) -> None:
    """Configure and launch the hub Uvicorn server.

    Parameters
    ----------
    runtime : HubRuntime
        Prepared runtime used to seed the controller and FastAPI app.
    """

    try:
        print_hub_startup_banner(runtime.config)
        uvconfig = setup_hub_server(runtime)
        server = uvicorn.Server(uvconfig)
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Hub server shutdown requested by user. Exiting...")
    except Exception:
        logger.exception("Hub server startup failed")
        sys.exit(1)
