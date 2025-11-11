"""Server setup helpers for the MLX OpenAI Server FastAPI application.

This module contains utilities to configure logging, create a FastAPI
lifespan context that initializes the appropriate MLX handler (LM, VLM,
Flux, Embeddings, Whisper), and to construct a :class:`uvicorn.Config`
that is ready to run the application.

The primary exported functions are:
- ``configure_logging``: configure loguru output to console and file
- ``create_lifespan``: return an asynccontextmanager that performs model
    initialization and cleanup during application startup/shutdown
- ``setup_server``: build the FastAPI app, attach middleware and
    exception handlers, and return a uvicorn config

These helpers keep the CLI and server wiring separated so the same
application behavior can be reused in different runners (CLI, tests,
or programmatic embedding).
"""

import gc
import time
from contextlib import asynccontextmanager
from typing import Any

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.endpoints import router
from app.handler import MFLUX_AVAILABLE, MLXFluxHandler
from app.handler.mlx_embeddings import MLXEmbeddingsHandler
from app.handler.mlx_lm import MLXLMHandler
from app.handler.mlx_vlm import MLXVLMHandler
from app.handler.mlx_whisper import MLXWhisperHandler
from app.version import __version__


def configure_logging(log_file=None, no_log_file=False, log_level="INFO"):
    """Configure loguru logging according to CLI parameters.

    This function removes the default loguru handler and adds a console
    handler with a human-friendly colored format. If file logging is
    enabled (``no_log_file`` is False), a rotating file handler is also
    added using ``log_file`` or the default ``logs/app.log`` path.

    Args:
        log_file (Optional[str]): Path to the log file. If ``None``, the
            default ``logs/app.log`` will be used when file logging is
            enabled.
        no_log_file (bool): If True, disable file logging entirely.
        log_level (str): Minimum log level for handlers (e.g. "INFO").
    """
    logger.remove()  # Remove default handler

    # Add console handler
    logger.add(
        lambda msg: print(msg),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "✦ <level>{message}</level>",
        colorize=True,
    )
    if not no_log_file:
        file_path = log_file if log_file else "logs/app.log"
        logger.add(
            file_path,
            rotation="500 MB",
            retention="10 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )


def get_model_identifier(args: Any) -> str:
    """Return the model identifier used to initialize handlers.

    The function centralizes any logic required to compute the value that
    identifies a model to the MLX handlers. Currently this is simply
    ``args.model_path``, but extracting it here keeps callers simple and
    makes future changes easier.

    Args:
        args: The configuration object returned by the CLI (typically a
            ``Config`` instance from ``app.cli``).

    Returns:
        str: The model identifier (path or name) used by handlers.
    """

    return args.model_path


def create_lifespan(config_args):
    """Create an async FastAPI lifespan context manager bound to configuration.

    The returned context manager performs the following actions during
    application startup:

    - Determine the model identifier from the provided ``config_args``
    - Instantiate the appropriate MLX handler based on ``model_type``
      (multimodal, image-generation, image-edit, embeddings, whisper, or
      text LM)
    - Initialize the handler (including queuing and concurrency setup)
    - Perform an initial memory cleanup

    During shutdown the lifespan will attempt to call the handler's
    ``cleanup`` method and perform final memory cleanup.

    Args:
        config_args: Object containing CLI configuration attributes used
            to initialize handlers (e.g., model_type, model_path,
            max_concurrency, queue_timeout, etc.).

    Returns:
        Callable: An asynccontextmanager usable as FastAPI ``lifespan``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Async context manager executed on application startup and shutdown.

        This inner function is the actual FastAPI lifespan callable. It
        initializes the selected MLX handler and attaches it to
        ``app.state.handler`` so request handlers can access it. It also
        sets up a basic memory-management pattern used for long-running
        processes (periodic calls to ``mx.clear_cache()`` and
        ``gc.collect()``).

        Args:
            app: The FastAPI application instance being started.
        """
        try:
            model_identifier = get_model_identifier(config_args)
            if config_args.model_type == "image-generation":
                logger.info(f"Initializing MLX handler with model name: {model_identifier}")
            else:
                logger.info(f"Initializing MLX handler with model path: {model_identifier}")

            if config_args.model_type == "multimodal":
                handler = MLXVLMHandler(
                    model_path=model_identifier,
                    context_length=getattr(config_args, "context_length", None),
                    max_concurrency=config_args.max_concurrency,
                    disable_auto_resize=getattr(config_args, "disable_auto_resize", False),
                    enable_auto_tool_choice=getattr(config_args, "enable_auto_tool_choice", False),
                    tool_call_parser=getattr(config_args, "tool_call_parser", None),
                    reasoning_parser=getattr(config_args, "reasoning_parser", None),
                )
            elif config_args.model_type == "image-generation":
                if not MFLUX_AVAILABLE:
                    raise ValueError(
                        "Image generation requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git"
                    )
                if config_args.config_name not in [
                    "flux-schnell",
                    "flux-dev",
                    "flux-krea-dev",
                ]:
                    raise ValueError(
                        f"Invalid config name: {config_args.config_name}. Only flux-schnell, flux-dev, and flux-krea-dev are supported for image generation."
                    )
                handler = MLXFluxHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                    quantize=getattr(config_args, "quantize", 8),
                    config_name=config_args.config_name,
                    lora_paths=getattr(config_args, "lora_paths", None),
                    lora_scales=getattr(config_args, "lora_scales", None),
                )
            elif config_args.model_type == "embeddings":
                handler = MLXEmbeddingsHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                )
            elif config_args.model_type == "image-edit":
                if not MFLUX_AVAILABLE:
                    raise ValueError(
                        "Image editing requires mflux. Install with: pip install git+https://github.com/cubist38/mflux.git"
                    )
                if config_args.config_name != "flux-kontext-dev":
                    raise ValueError(
                        f"Invalid config name: {config_args.config_name}. Only flux-kontext-dev is supported for image edit."
                    )
                handler = MLXFluxHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                    quantize=getattr(config_args, "quantize", 8),
                    config_name=config_args.config_name,
                    lora_paths=getattr(config_args, "lora_paths", None),
                    lora_scales=getattr(config_args, "lora_scales", None),
                )
            elif config_args.model_type == "whisper":
                handler = MLXWhisperHandler(
                    model_path=model_identifier,
                    max_concurrency=config_args.max_concurrency,
                )
            else:
                handler = MLXLMHandler(
                    model_path=model_identifier,
                    context_length=getattr(config_args, "context_length", None),
                    max_concurrency=config_args.max_concurrency,
                    enable_auto_tool_choice=getattr(config_args, "enable_auto_tool_choice", False),
                    tool_call_parser=getattr(config_args, "tool_call_parser", None),
                    reasoning_parser=getattr(config_args, "reasoning_parser", None),
                )
            # Initialize queue
            await handler.initialize(
                {
                    "max_concurrency": config_args.max_concurrency,
                    "timeout": config_args.queue_timeout,
                    "queue_size": config_args.queue_size,
                }
            )
            logger.info("MLX handler initialized successfully")
            app.state.handler = handler

        except Exception as e:
            logger.error(f"Failed to initialize MLX handler: {str(e)}")
            raise

        # Initial memory cleanup
        mx.clear_cache()
        gc.collect()

        yield

        # Shutdown
        logger.info("Shutting down application")
        if hasattr(app.state, "handler") and app.state.handler:
            try:
                logger.info("Cleaning up resources")
                await app.state.handler.cleanup()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during shutdown: {str(e)}")

        # Final memory cleanup
        mx.clear_cache()
        gc.collect()

    return lifespan


# App instance will be created during setup with the correct lifespan
app = None


def setup_server(args) -> uvicorn.Config:
    global app

    """Create and configure the FastAPI app and return a Uvicorn config.

    This function sets up logging, constructs the FastAPI application with
    a configured lifespan, registers routes and middleware, and returns a
    :class:`uvicorn.Config` ready to be used to run the server.

    Note: This function mutates the module-level ``app`` global variable.

    Args:
        args: Configuration object usually produced by the CLI. Expected
            to have attributes like ``host``, ``port``, ``log_level``,
            and logging-related fields.

    Returns:
        uvicorn.Config: A configuration object that can be passed to
        ``uvicorn.Server(config).run()`` to start the application.
    """

    # Configure logging based on CLI parameters
    configure_logging(
        log_file=getattr(args, "log_file", None),
        no_log_file=getattr(args, "no_log_file", False),
        log_level=getattr(args, "log_level", "INFO"),
    )

    # Create FastAPI app with the configured lifespan
    app = FastAPI(
        title="OpenAI-compatible API",
        description="API for OpenAI-compatible chat completion and text embedding",
        version=__version__,
        lifespan=create_lifespan(args),
    )

    app.include_router(router)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Middleware to add processing time header and run cleanup.

        Measures request processing time, appends an ``X-Process-Time``
        header, and increments a simple request counter used to trigger
        periodic memory cleanup for long-running processes.
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Periodic memory cleanup for long-running processes
        if hasattr(request.app.state, "request_count"):
            request.app.state.request_count += 1
        else:
            request.app.state.request_count = 1

        # Clean up memory every 50 requests
        if request.app.state.request_count % 50 == 0:
            mx.clear_cache()
            gc.collect()
            logger.debug(
                f"Performed memory cleanup after {request.app.state.request_count} requests"
            )

        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler that logs and returns a 500 payload.

        Logs the exception (with traceback) and returns a generic JSON
        response with a 500 status code so internal errors do not leak
        implementation details to clients.
        """
        logger.error(f"Global exception handler caught: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    logger.info(f"Starting server on {args.host}:{args.port}")
    config = uvicorn.Config(
        app=app, host=args.host, port=args.port, log_level=args.log_level.lower(), access_log=True
    )
    return config
