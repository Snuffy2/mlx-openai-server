"""Command-line interface and helpers for the MLX server.

This module defines the Click command group used by the package and the
``launch`` command which constructs a server configuration and starts
the ASGI server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import IO, Any, Literal
from urllib.parse import quote

import click
import httpx
from loguru import logger

from .config import MLXServerConfig
from .const import (
    DEFAULT_BIND_HOST,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HUB_CONFIG_PATH,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PORT,
    DEFAULT_QUANTIZE,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_QUEUE_TIMEOUT,
)
from .handler.parser.factory import PARSER_REGISTRY
from .hub.config import HubConfigError, MLXHubConfig, load_hub_config

# Hub IPC service removed: CLI uses HTTP API to contact the hub daemon
from .main import start
from .version import __version__


class UpperChoice(click.Choice[str]):
    """Case-insensitive choice type that returns canonical, uppercase values.

    This convenience subclass normalizes user input in a case-insensitive way
    but returns the canonical, uppercase option value from ``self.choices``. It is useful
    for flags like ``--log-level`` where callers expect the stored value to
    exactly match one of the declared choices.
    """

    def normalize_choice(self, choice: str | None, ctx: click.Context | None) -> str | None:  # type: ignore[override]
        """
        Normalize a user's choice to the canonical option using case-insensitive matching.

        Parameters:
            choice (str | None): User-supplied value; may be None.
            ctx (click.Context | None): Click context object (unused).

        Returns:
            str | None: The canonical option string from `self.choices` that matches `choice` (preserving the original choice casing from `self.choices`), or `None` if `choice` is None.
        """
        if choice is None:
            return None
        upperchoice = choice.upper()
        for opt in self.choices:
            if opt.upper() == upperchoice:
                return opt  # return the canonical opt
        self.fail(
            f"Invalid choice: {choice}. (choose from {', '.join(self.choices)})",
            param=None,
            ctx=ctx,
        )
        return None


# Configure basic logging for CLI (will be overridden by main.py)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "✦ <level>{message}</level>",
    colorize=True,
    level="INFO",
)


@click.group()
@click.version_option(
    version=__version__,
    message="""
✨ %(prog)s - OpenAI Compatible API Server for MLX models ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Version: %(version)s
""",
)
def cli() -> None:
    """Top-level Click command group for the MLX server CLI.

    Subcommands (such as ``launch``) are registered on this group and
    invoked by the console entry point.
    """


def _load_hub_config_or_fail(config_path: str | None) -> MLXHubConfig:
    """Load hub configuration or exit with a CLI error.

    Parameters
    ----------
    config_path : str or None
        Path to the hub config file.

    Returns
    -------
    MLXHubConfig
        Loaded hub configuration.

    Raises
    ------
    click.ClickException
        If the configuration cannot be loaded.
    """
    try:
        return load_hub_config(config_path)
    except HubConfigError as exc:  # pragma: no cover - CLI friendly errors
        raise click.ClickException(str(exc)) from exc


def _controller_base_url(config: MLXHubConfig) -> str:
    """
    Determine the hub daemon's base HTTP URL, preferring a live runtime state when available.

    When a transient runtime state file exists and contains host/port values, those are used (with validation of the port); otherwise the values from `config` or sensible defaults are used.

    Returns:
        The base HTTP URL for the hub daemon, e.g. "http://host:port".
    """
    # Prefer runtime state file (written by `hub start`) when available
    runtime = _read_hub_runtime_state(config)
    if runtime:
        host = runtime.get("host") or (config.host or DEFAULT_BIND_HOST)
        # runtime may contain untyped values (loaded from JSON). Validate
        # the port before converting to int to keep mypy and runtime checks happy.
        rt_port = runtime.get("port")
        if isinstance(rt_port, (int, str)):
            try:
                port = int(rt_port)
            except Exception:
                port = int(config.port or DEFAULT_PORT)
        else:
            port = int(config.port or DEFAULT_PORT)
        return f"http://{host}:{port}"

    host = config.host or DEFAULT_BIND_HOST
    port = config.port
    return f"http://{host}:{port}"


def _runtime_state_path(config: MLXHubConfig) -> Path:
    """
    Get the path to the transient hub runtime state file.

    Chooses the configured log directory when available; otherwise falls back to a "logs"
    directory in the current working directory and returns the full Path to "hub_runtime.json".

    Returns:
        Path: Full path to the hub runtime state file.
    """
    try:
        log_dir = (
            Path(config.log_path) if getattr(config, "log_path", None) else Path.cwd() / "logs"
        )
    except Exception:
        log_dir = Path.cwd() / "logs"
    return log_dir / "hub_runtime.json"


def _write_hub_runtime_state(config: MLXHubConfig, pid: int) -> None:
    """
    Persist transient hub runtime metadata to the hub runtime state file.

    Writes a small JSON payload containing `pid`, `host`, `port`, and `started_at` to the runtime state file used by other CLI commands to locate a running daemon. The target path is derived from the provided `config` (typically under the configured logs directory). This operation is best-effort: it logs success on write and logs a warning on failure without raising exceptions.

    Parameters:
        config (MLXHubConfig): Hub configuration used to determine host, port, and runtime-state path.
        pid (int): Process ID of the running hub daemon to record.
    """
    path = _runtime_state_path(config)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": int(pid),
            "host": config.host or DEFAULT_BIND_HOST,
            "port": int(config.port or DEFAULT_PORT),
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        path.write_text(json.dumps(payload))
        logger.debug(f"Wrote hub runtime state to {path}")
    except Exception as e:  # pragma: no cover - best-effort logging
        logger.warning(f"Failed to write hub runtime state to {path}. {type(e).__name__}: {e}")


def _read_hub_runtime_state(config: MLXHubConfig) -> dict[str, object] | None:
    """
    Determine the persisted hub runtime state and return its PID, host, and port if the referenced process appears to be running.

    Performs a best-effort liveness check of the PID stored in the runtime state file.

    Returns:
        dict: A dictionary with keys `'pid'` (int), `'host'` (str), and `'port'` (int) when a valid runtime state exists and the process is alive, `None` otherwise.
    """
    path = _runtime_state_path(config)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        pid = int(data.get("pid"))
        host = data.get("host") or DEFAULT_BIND_HOST
        raw_port = data.get("port")
        if raw_port is None:
            port = int(config.port or DEFAULT_PORT)
        else:
            try:
                port = int(raw_port)
            except Exception:
                port = int(config.port or DEFAULT_PORT)
    except Exception:
        return None

    # Check PID alive (best-effort)
    pid_alive = False
    try:
        # os.kill with signal 0 raises OSError if process does not exist
        os.kill(pid, 0)
        pid_alive = True
    except Exception:
        pid_alive = False

    if not pid_alive:
        return None

    return {"pid": pid, "host": host, "port": port}


def _call_daemon_api(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 5.0,
) -> dict[str, object] | None:
    """
    Call the hub daemon HTTP API and return its parsed JSON payload.

    Parameters:
        config (MLXHubConfig): Hub configuration used to determine the daemon base URL.
        method (str): HTTP method (e.g., "GET", "POST").
        path (str): Request path starting with '/' to append to the base URL.
        json (object | None): JSON body to include with the request, if any.
        timeout (float): Request timeout in seconds.

    Returns:
        dict[str, object] | None: Parsed JSON object from the response, `None` if the response has no content, or `{"raw": "<text>"}` if the body is non-JSON.

    Raises:
        click.ClickException: If the daemon is unreachable or responds with a 4xx/5xx status.
    """
    base = _controller_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except httpx.HTTPError as e:  # pragma: no cover - network error handling
        raise click.ClickException(f"Failed to contact hub daemon at {base}: {e}") from e

    if resp.status_code >= 400:
        # Try to include JSON error message if present
        try:
            payload: object = resp.json()
        except ValueError:
            payload = resp.text
        raise click.ClickException(f"Daemon responded {resp.status_code}: {payload}")

    if not resp.content:
        return None

    try:
        payload = resp.json()
    except ValueError:
        return {"raw": resp.text}

    # Ensure we return a dict[str, object] as declared; if the JSON is not a
    # mapping, fall back to returning the raw text.
    if isinstance(payload, dict):
        return payload
    return {"raw": resp.text}


def _print_hub_status(
    config: MLXHubConfig,
    *,
    model_names: Iterable[str] | None = None,
    live_status: dict[str, Any] | None = None,
) -> None:
    """
    Print a formatted table of configured hub models and their runtime state.

    Prints the hub log path and whether the status page is enabled, then lists configured models (filtered by `model_names` when provided) with columns NAME, STATE, LOADED, AUTO-UNLOAD, TYPE, GROUP, DEFAULT, and MODEL. When `live_status` is supplied, runtime metadata (such as process state, pid, port, and memory state) is used to enrich the displayed STATE and LOADED values.

    Parameters:
        config (MLXHubConfig): The hub configuration containing models and display settings.
        model_names (Iterable[str] | None, optional): If provided, only models whose names appear in this iterable are displayed (blank names are ignored).
        live_status (dict[str, Any] | None, optional): Live status payload from the hub service; expected to contain a "models" list with per-model metadata used to show runtime state.
    """
    click.echo(f"Hub log path: {config.log_path}")
    click.echo(f"Status page enabled: {'yes' if config.enable_status_page else 'no'}")

    selection = None
    if model_names:
        selection = {name.strip() for name in model_names if name.strip()}
    configured = []
    for model in config.models:
        if selection and model.name not in selection:
            continue
        configured.append(model)

    if not configured:
        click.echo("No matching models in hub config")
        return

    live_lookup: dict[str, dict[str, Any]] = {}
    if live_status:
        for entry in live_status.get("models", []):
            name = entry.get("id")  # Model objects have "id"
            if isinstance(name, str):
                live_lookup[name] = entry

    click.echo("Models:")
    headers = ["NAME", "STATE", "LOADED", "AUTO-UNLOAD", "TYPE", "GROUP", "DEFAULT", "MODEL"]
    rows: list[dict[str, str]] = []
    for model in configured:
        name = model.name or "<unnamed>"
        live = live_lookup.get(name)
        metadata = (live or {}).get("metadata", {})
        state = metadata.get("process_state", "inactive")
        pid = metadata.get("pid")
        port = metadata.get("port") or model.port

        # Format state with pid and port if running
        if state == "running" and pid is not None:
            state_display = f"{state} (pid={pid}"
            if port is not None:
                state_display += f", port={port}"
            state_display += ")"
        else:
            state_display = state

        # Loaded in memory: prefer explicit runtime flag when available,
        # otherwise approximate with process state.
        memory_flag = metadata.get("memory_state") == "loaded"
        loaded_in_memory = "yes" if memory_flag else "no"

        # Auto-unload
        auto_unload = f"{model.auto_unload_minutes}min" if model.auto_unload_minutes else "-"

        # Model type
        model_type = model.model_type

        # Group
        group = model.group or "-"

        # Default
        default = "✓" if model.is_default_model else "-"

        # Model path
        model_path = model.model_path

        rows.append(
            {
                "NAME": name,
                "STATE": state_display,
                "LOADED": loaded_in_memory,
                "AUTO-UNLOAD": auto_unload,
                "TYPE": model_type,
                "GROUP": group,
                "DEFAULT": default,
                "MODEL": model_path,
            },
        )

    # Calculate column widths
    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    # Print header
    header_line = "  " + " | ".join(header.ljust(widths[header]) for header in headers)
    click.echo(header_line)

    # Print divider
    divider = "  " + "-+-".join("-" * widths[header] for header in headers)
    click.echo(divider)

    # Print rows
    for row in rows:
        row_line = "  " + " | ".join(row[header].ljust(widths[header]) for header in headers)
        click.echo(row_line)


_FLASH_STYLES: dict[str, tuple[str, str]] = {
    "info": ("[info]", "cyan"),
    "success": ("[ok]", "green"),
    "warning": ("[warn]", "yellow"),
    "error": ("[err]", "red"),
}


def _flash(message: str, tone: Literal["info", "success", "warning", "error"] = "info") -> None:
    """
    Emit a short, colorized status line with a tone-specific prefix.

    Parameters:
        message (str): Text to display.
        tone (Literal["info", "success", "warning", "error"]): One of four display tones that selects the prefix and color; defaults to "info".
    """
    prefix, color = _FLASH_STYLES.get(tone, _FLASH_STYLES["info"])
    click.echo(click.style(f"{prefix} {message}", fg=color))


def _format_name_list(values: Iterable[str] | None) -> str:
    """Format a list of names into a comma-separated string.

    Parameters
    ----------
    values : Iterable[str] | None
        The list of names to format.

    Returns
    -------
    str
        Comma-separated string of names, or "none" if empty.
    """
    if not values:
        return "none"
    filtered = [value for value in values if value]
    return ", ".join(filtered) if filtered else "none"


def _emit_reload_summary(diff: dict[str, Any], *, header: str) -> None:
    """
    Emit a concise reload summary line to the console.

    Formats the lists of started, stopped, and unchanged models from the provided diff and emits a single status line prefixed by `header`. Uses a `success` tone if any models started or stopped, otherwise `info`.

    Parameters:
        diff (dict[str, Any]): Reload diff containing keys "started", "stopped", and "unchanged" with iterables of model names.
        header (str): Leading text to prefix the summary line.
    """
    started = _format_name_list(diff.get("started"))
    stopped = _format_name_list(diff.get("stopped"))
    unchanged = _format_name_list(diff.get("unchanged"))
    tone: Literal["info", "success"] = (
        "success" if started != "none" or stopped != "none" else "info"
    )
    _flash(f"{header}: started={started} | stopped={stopped} | unchanged={unchanged}", tone=tone)


def _reload_or_fail(config: MLXHubConfig, *, header: str) -> dict[str, Any]:
    """Reload the hub daemon and emit a summary, or fail with an exception.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration used to contact the daemon.
    header : str
        The header message for the reload summary.

    Returns
    -------
    dict[str, Any]
        The reload diff.

    Raises
    ------
    click.ClickException
        If the reload operation fails.
    """
    try:
        diff = _call_daemon_api(config, "POST", "/hub/reload") or {}
    except click.ClickException as e:
        raise click.ClickException(f"Hub reload failed: {e}") from e
    _emit_reload_summary(diff, header=header)
    return diff


def _require_service_client(config: MLXHubConfig) -> bool:
    """
    Verify that the hub daemon is reachable and responsive.

    Parameters:
        config (MLXHubConfig): Hub configuration used to locate the daemon.

    Returns:
        True if the daemon is reachable.

    Raises:
        click.ClickException: If the hub manager is not running.
    """
    try:
        _call_daemon_api(config, "GET", "/health", timeout=1.0)
    except click.ClickException as exc:
        raise click.ClickException(
            "Hub manager is not running. Start it via 'mlx-openai-server hub start'.",
        ) from exc
    return True


def _perform_memory_action_request(
    config: MLXHubConfig,
    model_name: str,
    action: Literal["load", "unload"],
) -> tuple[bool, str]:
    """
    Request a memory `load` or `unload` action for a model from the hub controller.

    If the controller returns a JSON payload containing a `message` field that will be used as the returned message; otherwise a default confirmation text is returned. If a connectivity or API error occurs, the function returns `False` with the error text.

    Returns:
        tuple[bool, str]: `success` is `True` if the request completed without raising a ClickException, `False` otherwise; `message` is the controller-provided message or an error description.
    """
    try:
        payload = _call_daemon_api(
            config,
            "POST",
            f"/hub/models/{quote(model_name, safe='')}/{action}",
            timeout=10.0,
        )
    except click.ClickException as exc:
        return False, str(exc)
    raw_message = (payload or {}).get("message")
    if raw_message is None:
        message = f"Memory {action} requested"
    else:
        message = str(raw_message)
    return True, message


def _run_memory_actions(
    config: MLXHubConfig,
    model_names: Iterable[str],
    action: Literal["load", "unload"],
) -> None:
    """
    Initiate memory load or unload requests for multiple models and report per-model results.

    Parameters:
        config (MLXHubConfig): Hub configuration used to contact the daemon.
        model_names (Iterable[str]): Iterable of model names; blank names are skipped with a warning.
        action (Literal["load", "unload"]): Memory action to request for each model.

    Raises:
        click.ClickException: If one or more memory actions fail.
    """
    had_error = False
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            had_error = True
            _flash("Skipping blank model name entry", tone="warning")
            continue
        ok, message = _perform_memory_action_request(config, target, action)
        verb = action
        if ok:
            _flash(f"{target}: memory {verb} requested ({message})", tone="success")
        else:
            had_error = True
            _flash(f"{target}: {message}", tone="error")
    if had_error:
        raise click.ClickException("One or more memory actions failed")


def _format_duration(seconds: float | None) -> str:
    """
    Format a duration in seconds into a compact human-readable string.

    If `seconds` is None or negative, returns "-". For durations of one hour or more the result is formatted as
    "{hours}h{minutes:02d}m"; for durations of one minute or more as "{minutes}m{seconds:02d}s"; otherwise as "{seconds}s".

    Parameters:
        seconds (float | None): Duration in seconds.

    Returns:
        str: Compact formatted duration.
    """
    if seconds is None or seconds < 0:
        return "-"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _render_watch_table(models: Iterable[dict[str, Any]], *, now: float | None = None) -> str:
    """
    Format a fixed-width table describing hub-managed processes.

    Parameters:
        models (Iterable[dict[str, Any]]): Iterable of model process records. Each record may include
            keys: "name", "state", "pid", "group", "started_at" (epoch seconds), "exit_code", and "log_path".
        now (float | None, optional): Reference UNIX timestamp used to compute UPTIME; if omitted the
            current time is used.

    Returns:
        str: Multi-line string containing a table with headers NAME, STATE, PID, GROUP, UPTIME, EXIT, LOG.
             If `models` is empty, returns the string "  (no managed processes)".
    """
    snapshot = list(models)
    if not snapshot:
        return "  (no managed processes)"

    reference = now if isinstance(now, (int, float)) else time.time()
    headers = ["NAME", "STATE", "PID", "GROUP", "UPTIME", "EXIT", "LOG"]
    rows: list[dict[str, str]] = []
    for entry in sorted(snapshot, key=lambda item: str(item.get("name", "?"))):
        name = str(entry.get("name", "?"))
        state = str(entry.get("state", "unknown")).upper()
        pid = str(entry.get("pid") or "-")
        group = entry.get("group") or "-"
        started_at = entry.get("started_at")
        uptime = "-"
        if isinstance(started_at, (int, float)):
            uptime = _format_duration(reference - float(started_at))
        exit_code = entry.get("exit_code")
        exit_display = "-" if exit_code in (None, 0) else str(exit_code)
        log_path = entry.get("log_path")
        log_display = Path(log_path).name if isinstance(log_path, str) else "-"
        rows.append(
            {
                "NAME": name,
                "STATE": state,
                "PID": pid,
                "GROUP": group,
                "UPTIME": uptime,
                "EXIT": exit_display,
                "LOG": log_display,
            },
        )

    widths: dict[str, int] = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    divider = "  " + "-+-".join("-" * widths[header] for header in headers)
    header_line = "  " + " | ".join(header.ljust(widths[header]) for header in headers)
    lines = [header_line, divider]
    lines.extend(
        "  " + " | ".join(row[header].ljust(widths[header]) for header in headers) for row in rows
    )
    return "\n".join(lines)


def _print_watch_snapshot(snapshot: dict[str, Any]) -> None:
    """
    Prints a timestamped status line and a formatted table describing hub-managed processes from a snapshot.

    The snapshot may include a "timestamp" (Unix seconds) used for the header and a "models" list of per-model dictionaries. The function prints a one-line summary with counts of total, running, stopped, and failed models, then prints a multi-line table rendered from the models list.

    Parameters:
        snapshot (dict[str, Any]): Snapshot data; expected keys:
            - "timestamp" (optional): numeric Unix timestamp used for the header if present.
            - "models" (optional): iterable of model info dictionaries to be rendered.
    """
    timestamp = snapshot.get("timestamp")
    reference = timestamp if isinstance(timestamp, (int, float)) else time.time()
    formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reference))

    raw_models = snapshot.get("models")
    models: list[dict[str, Any]] = raw_models if isinstance(raw_models, list) else []
    running = sum(1 for entry in models if str(entry.get("state")).lower() == "running")
    stopped = sum(1 for entry in models if str(entry.get("state")).lower() == "stopped")
    failed = sum(
        1
        for entry in models
        if str(entry.get("state")).lower() == "failed" or entry.get("exit_code") not in (None, 0)
    )

    click.echo(
        f"[{formatted}] models={len(models)} running={running} stopped={stopped} failed={failed}",
    )
    click.echo(_render_watch_table(models, now=reference))


@cli.command(help="Start the MLX OpenAI Server with the supplied flags")
@click.option(
    "--model-path",
    required=True,
    type=str,
    help="Path to the model. Accepts local paths or Hugging Face repository IDs (e.g., 'blackforestlabs/FLUX.1-dev').",
)
@click.option(
    "--model-type",
    default=DEFAULT_MODEL_TYPE,
    type=click.Choice(
        ["lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper"],
    ),
    help="Type of model to run (lm: text-only, multimodal: text+vision+audio, image-generation: flux image generation, image-edit: flux image edit, embeddings: text embeddings, whisper: audio transcription)",
)
@click.option(
    "--context-length",
    default=DEFAULT_CONTEXT_LENGTH,
    type=int,
    help="Context length for language models. Only works with `lm` or `multimodal` model types.",
)
@click.option("--port", default=DEFAULT_PORT, type=int, help="Port to run the server on")
@click.option("--host", default=DEFAULT_BIND_HOST, help="Host to run the server on")
@click.option(
    "--max-concurrency",
    default=DEFAULT_MAX_CONCURRENCY,
    type=int,
    help="Maximum number of concurrent requests",
)
@click.option(
    "--queue-timeout",
    default=DEFAULT_QUEUE_TIMEOUT,
    type=int,
    help="Request timeout in seconds",
)
@click.option(
    "--queue-size",
    default=DEFAULT_QUEUE_SIZE,
    type=int,
    help="Maximum queue size for pending requests",
)
@click.option(
    "--quantize",
    default=DEFAULT_QUANTIZE,
    type=int,
    help="Quantization level for the model. Only used for image-generation and image-edit Flux models.",
)
@click.option(
    "--config-name",
    default=None,
    type=click.Choice(["flux-schnell", "flux-dev", "flux-krea-dev", "flux-kontext-dev"]),
    help="Config name of the model. Only used for image-generation and image-edit Flux models.",
)
@click.option(
    "--lora-paths",
    default=None,
    type=str,
    help="Path to the LoRA file(s). Multiple paths should be separated by commas.",
)
@click.option(
    "--lora-scales",
    default=None,
    type=str,
    help="Scale factor for the LoRA file(s). Multiple scales should be separated by commas.",
)
@click.option(
    "--disable-auto-resize",
    is_flag=True,
    help="Disable automatic model resizing. Only work for Vision Language Models.",
)
@click.option(
    "--log-file",
    default=None,
    type=str,
    help="Path to log file. If not specified, logs will be written to 'logs/app.log' by default.",
)
@click.option(
    "--no-log-file",
    is_flag=True,
    help="Disable file logging entirely. Only console output will be shown.",
)
@click.option(
    "--log-level",
    default=DEFAULT_LOG_LEVEL,
    type=UpperChoice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set the logging level. Default is INFO.",
)
@click.option(
    "--enable-auto-tool-choice",
    is_flag=True,
    help="Enable automatic tool choice. Only works with language models.",
)
@click.option(
    "--tool-call-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify tool call parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--reasoning-parser",
    default=None,
    type=click.Choice(list(PARSER_REGISTRY.keys())),
    help="Specify reasoning parser to use instead of auto-detection. Only works with language models.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Enable trust_remote_code when loading models. This allows loading custom code from model repositories.",
)
@click.option(
    "--jit",
    "jit_enabled",
    is_flag=True,
    help="Enable just-in-time model loading. Models load on first request instead of startup.",
)
@click.option(
    "--auto-unload-minutes",
    type=click.IntRange(1),
    default=None,
    help="When JIT is enabled, unload the model after idle for this many minutes.",
)
def launch(
    model_path: str,
    model_type: str,
    context_length: int,
    port: int,
    host: str,
    max_concurrency: int,
    queue_timeout: int,
    queue_size: int,
    quantize: int,
    config_name: str | None,
    lora_paths: str | None,
    lora_scales: str | None,
    disable_auto_resize: bool,
    log_file: str | None,
    no_log_file: bool,
    log_level: str,
    enable_auto_tool_choice: bool,
    tool_call_parser: str | None,
    reasoning_parser: str | None,
    trust_remote_code: bool,
    jit_enabled: bool,
    auto_unload_minutes: int | None,
) -> None:
    """
    Start the single-model FastAPI/Uvicorn server using the provided configuration.

    Constructs an MLXServerConfig from the arguments and runs the server lifecycle.

    Parameters:
        model_type (str):
            Logical model category (e.g. "lm", "multimodal", "image-generation", "image-edit", "embeddings", "whisper").
        quantize (int):
            Quantization level; applicable to Flux image-generation and image-edit models.
        config_name (str | None):
            Model configuration name for Flux image-generation and image-edit models.
        lora_paths (str | None):
            Comma-separated path(s) to LoRA file(s) to apply to the model.
        lora_scales (str | None):
            Comma-separated scale(s) corresponding to `lora_paths`.
        auto_unload_minutes (int | None):
            Idle minutes before automatic model unload; only valid when JIT is enabled.

    Raises:
        click.BadOptionUsage:
            If `auto_unload_minutes` is provided while JIT is not enabled.
    """
    if auto_unload_minutes is not None and not jit_enabled:
        raise click.BadOptionUsage(
            "--auto-unload-minutes",
            "--auto-unload-minutes requires --jit to be set.",
        )

    # Validate model_path at runtime and provide a clear CLI error if missing.
    if model_path is None:
        logger.error("launch: missing required parameter 'model_path'")
        raise click.ClickException("Missing required argument: --model-path")

    args = MLXServerConfig(
        model_path=model_path,
        model_type=model_type,
        context_length=context_length,
        port=port,
        host=host,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
        queue_size=queue_size,
        quantize=quantize,
        config_name=config_name,
        lora_paths_str=lora_paths,
        lora_scales_str=lora_scales,
        disable_auto_resize=disable_auto_resize,
        log_file=log_file,
        no_log_file=no_log_file,
        log_level=log_level,
        enable_auto_tool_choice=enable_auto_tool_choice,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        trust_remote_code=trust_remote_code,
        jit_enabled=jit_enabled,
        auto_unload_minutes=auto_unload_minutes,
    )

    asyncio.run(start(args))


@cli.group(help="Manage hub-based multi-model deployments", invoke_without_command=True)
@click.option(
    "--config",
    "hub_config_path",
    default=None,
    help=f"Path to hub YAML (default: {DEFAULT_HUB_CONFIG_PATH})",
)
@click.pass_context
def hub(
    ctx: click.Context,
    hub_config_path: str | None,
) -> None:
    """Entry point for hub sub-commands."""
    ctx.ensure_object(dict)
    ctx.obj["hub_config_path"] = hub_config_path
    if ctx.invoked_subcommand is None:
        ctx.invoke(hub_start)


def _start_hub_daemon(config: MLXHubConfig) -> subprocess.Popen[bytes] | None:
    """
    Start the hub daemon subprocess if no running daemon is detected.

    If a hub manager is already reachable via its health endpoint, no action is taken.

    Returns:
        The started subprocess.Popen[bytes] instance when a new daemon was launched, or `None` if a daemon was already running.

    Raises:
        click.ClickException: If the hub configuration is not saved to disk or the daemon fails to start or become healthy within the startup timeout.
    """
    # Check daemon availability
    try:
        _call_daemon_api(config, "GET", "/health", timeout=2.0)
    except click.ClickException:
        pass  # Not running, proceed to start
    else:
        return None  # Already running

    if config.source_path is None:
        raise click.ClickException(
            "Hub configuration must be saved to disk before starting the manager.",
        )

    click.echo("Starting hub manager...")
    host_val = config.host or DEFAULT_BIND_HOST
    port_val = str(config.port)

    # Set environment variable for daemon to use the same config
    env = os.environ.copy()
    if config.source_path:
        env["MLX_HUB_CONFIG_PATH"] = str(config.source_path)

    cmd = [
        sys.executable,  # Use the same Python executable
        "-m",
        "uvicorn",
        "app.hub.daemon:create_app",
        "--factory",
        "--host",
        host_val,
        "--port",
        port_val,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=Path.cwd(),
        )

        # Start background threads to log subprocess output
        def _log_output(stream: IO[bytes], level: str, prefix: str) -> None:
            """
            Read lines from a subprocess binary stream, decode them, and emit each non-empty line to the logger with a prefixed log message at the given level.

            Parameters:
                stream (IO[bytes]): Binary stream to read lines from (e.g., subprocess stdout or stderr).
                level (str): Log level to use for each line; expected values include "info", "error", or others for debug.
                prefix (str): Text prefix included in every logged message to identify the stream source.

            Notes:
                Any error encountered while reading the stream is caught and emitted as a logger warning.
            """
            try:
                for line in iter(stream.readline, b""):
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    if line_str:
                        if level == "info":
                            logger.info(f"{prefix}: {line_str}")
                        elif level == "error":
                            logger.error(f"{prefix}: {line_str}")
                        else:
                            logger.debug(f"{prefix}: {line_str}")
            except Exception as e:
                logger.warning(f"Error reading subprocess {prefix} output: {e}")

        # Start threads to read stdout and stderr
        if proc.stdout:
            stdout_thread = threading.Thread(
                target=_log_output,
                args=(proc.stdout, "info", f"hub-daemon[{proc.pid}].stdout"),
                daemon=True,
            )
            stdout_thread.start()

        if proc.stderr:
            stderr_thread = threading.Thread(
                target=_log_output,
                args=(
                    proc.stderr,
                    "info",
                    f"hub-daemon[{proc.pid}].stderr",
                ),  # MLX-LM outputs to stderr so treat as info
                daemon=True,
            )
            stderr_thread.start()

        click.echo(f"Hub manager process started (PID: {proc.pid})")
    except Exception as e:
        raise click.ClickException(f"Failed to start hub manager: {e}") from e

    # Wait for daemon to become available
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            _call_daemon_api(config, "GET", "/health", timeout=1.0)
            click.echo("Hub manager is now running.")
            break
        except click.ClickException:
            time.sleep(0.5)
    else:
        raise click.ClickException(
            "Hub manager failed to start within 20 seconds.\n"
            "You can also start it manually (for example):\n"
            f"  uvicorn app.hub.daemon:create_app --host {host_val} --port {port_val}",
        )

    return proc


def _auto_start_default_models(config: MLXHubConfig) -> None:
    """
    Start models marked as default in the hub configuration by requesting the controller to start their processes.

    Refreshes the controller state before taking action. For each configured model whose `is_default_model` flag is true, requests a process start; if the start request fails for a model that is not JIT-enabled, requests a memory load as a fallback. Emits user-facing status messages and logs non-fatal errors while continuing with other models.

    Parameters:
        config (MLXHubConfig): Hub configuration containing model definitions.
    """
    try:
        # Refresh the controller state so it sees the latest config
        _call_daemon_api(config, "POST", "/hub/reload")
        for model in config.models:
            try:
                if not getattr(model, "is_default_model", False):
                    continue
                name = model.name
                jit = bool(getattr(model, "jit_enabled", False))
                click.echo(f"Requesting process start for default model: {name}")
                try:
                    _call_daemon_api(
                        config,
                        "POST",
                        f"/hub/models/{quote(str(name), safe='')}/start",
                    )
                    _flash(f"{name}: start requested", tone="success")
                except click.ClickException as exc_start:
                    # If start failed and the model is non-JIT, fall back
                    # to requesting a memory load so the configured default
                    # ends up available in the controller view.
                    if not jit:
                        try:
                            _call_daemon_api(
                                config,
                                "POST",
                                f"/hub/models/{quote(str(name), safe='')}/load",
                            )
                            _flash(f"{name}: memory load requested (fallback)", tone="success")
                        except click.ClickException as exc_load:
                            _flash(f"{name}: load failed ({exc_load})", tone="error")
                    else:
                        _flash(f"{name}: start failed ({exc_start})", tone="error")
            except Exception as e:  # pragma: no cover - best-effort
                logger.debug(f"Error while auto-starting default model. {type(e).__name__}: {e}")
    except Exception as e:
        # Ignore failures here; user can start models manually
        logger.exception("Failed to auto-start default models, continuing anyway", exc_info=e)


@hub.command(name="start", help="Start the hub manager ")
@click.argument("model_names", nargs=-1)
@click.pass_context
def hub_start(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Start the hub daemon (if not already running), persist transient runtime state when started, attempt to auto-start configured default models, and print the hub status and status-page URL.

    Parameters:
        model_names (tuple[str, ...]): If provided, restricts which models are shown in the printed status; pass an empty tuple or None to show all models.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    models = model_names or None

    proc = _start_hub_daemon(config)

    if proc is not None:
        # Persist runtime state for other CLI invocations to find this daemon
        try:
            _write_hub_runtime_state(config, proc.pid)
        except Exception:
            # Best-effort; do not fail start if writing runtime state fails
            logger.debug("Failed to write hub runtime state after start")

        _auto_start_default_models(config)

    click.echo(f"Status page enabled: {'yes' if config.enable_status_page else 'no'}")
    if config.enable_status_page:
        host_display = "localhost" if config.host == "0.0.0.0" else config.host
        click.echo(f"Browse to http://{host_display}:{config.port}/hub for the status dashboard")

    snapshot = None
    try:
        snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
    except click.ClickException as e:
        _flash(f"Unable to fetch live status: {e}", tone="warning")
    _print_hub_status(config, model_names=models, live_status=snapshot)


@hub.command(name="status", help="Show hub configuration and running processes")
@click.argument("model_names", nargs=-1)
@click.pass_context
def hub_status(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Display configured models and any active hub-managed processes.

    If the hub daemon is reachable, fetches live status and prints a combined view; if unreachable, prints configured models and emits a warning.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    snapshot = None
    try:
        snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
    except click.ClickException:
        _flash("Hub daemon is not running", tone="warning")
    _print_hub_status(config, model_names=model_names or None, live_status=snapshot)


@hub.command(name="reload", help="Reload hub.yaml and reconcile model processes")
@click.pass_context
def hub_reload(ctx: click.Context) -> None:
    """Force the running hub manager to reload its configuration.

    Parameters
    ----------
    ctx : click.Context
        Click context.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    try:
        diff = _call_daemon_api(config, "POST", "/hub/reload") or {}
    except click.ClickException as e:
        raise click.ClickException(f"Hub reload failed: {e}") from e
    _emit_reload_summary(diff, header="Hub reload complete")


@hub.command(name="stop", help="Stop the hub manager and all models")
@click.pass_context
def hub_stop(ctx: click.Context) -> None:
    """
    Request the hub manager to shut down and stop all managed models.

    If the hub daemon is not reachable, the command reports that nothing is running and exits without error.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        # If we can't contact the daemon it's likely not running; be friendly
        msg = str(e)
        if "Failed to contact hub daemon" in msg or "Daemon responded" in msg:
            _flash("Hub manager is not running; nothing to stop", tone="info")
            return
        raise click.ClickException(f"Config sync failed before shutdown: {e}") from e

    try:
        _call_daemon_api(config, "POST", "/hub/shutdown")
    except click.ClickException as e:
        msg = str(e)
        if "Failed to contact hub daemon" in msg:
            _flash("Hub manager is not running; nothing to stop", tone="info")
            return
        raise click.ClickException(f"Hub shutdown failed: {e}") from e

    _flash("Hub manager shutdown requested", tone="success")


@hub.command(name="start-model", help="Start one or more model processes")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_start_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Trigger start requests for one or more hub-managed models.

    The command reloads the hub configuration and then issues a start request for each non-blank model name, emitting a success or error flash per model.

    Parameters:
        model_names (tuple[str, ...]): Iterable of model names to start; blank entries are skipped.

    Raises:
        click.UsageError: If no model names are provided or all provided names are blank.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    if not model_names or all(not str(n).strip() for n in model_names):
        raise click.UsageError("Missing argument 'MODEL_NAMES'.")
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        raise click.ClickException(f"Config sync failed before load: {e}") from e
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            _call_daemon_api(config, "POST", f"/hub/models/{quote(target, safe='')}/start")
        except click.ClickException as e:
            _flash(f"{target}: start failed ({e})", tone="error")
        else:
            _flash(f"{target}: start requested", tone="success")


@hub.command(name="stop-model", help="Stop one or more model processes")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_stop_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Request the hub daemon to stop the named models.

    Synchronizes the hub configuration before sending a stop request for each provided model name. Raises a UsageError if no model names are provided. Blank model name entries are skipped with a warning; on config-sync failure a ClickException is raised. Per-model results are emitted as CLI status messages.

    Parameters:
        model_names (tuple[str, ...]): Tuple of model names to stop; whitespace-only names are ignored.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    if not model_names or all(not str(n).strip() for n in model_names):
        raise click.UsageError("Missing argument 'MODEL_NAMES'.")
    try:
        _call_daemon_api(config, "POST", "/hub/reload")
    except click.ClickException as e:
        raise click.ClickException(f"Config sync failed before unload: {e}") from e
    for raw_name in model_names:
        target = raw_name.strip()
        if not target:
            _flash("Skipping blank model name entry", tone="warning")
            continue
        try:
            _call_daemon_api(config, "POST", f"/hub/models/{quote(target, safe='')}/stop")
        except click.ClickException as e:
            _flash(f"{target}: stop failed ({e})", tone="error")
        else:
            _flash(f"{target}: stop requested", tone="success")


@hub.command(name="load-model", help="Load handlers for one or more models into memory")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_load_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Trigger memory load requests for each model in `model_names` via the hub controller.

    Parameters:
        model_names (tuple[str, ...]): Model names to load; blank or empty names are skipped.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "load")


@hub.command(name="unload-model", help="Unload handlers for one or more models from memory")
@click.argument("model_names", nargs=-1, required=True)
@click.pass_context
def hub_unload_model(ctx: click.Context, model_names: tuple[str, ...]) -> None:
    """
    Unload the specified models from memory using the hub controller.

    Parameters:
        model_names (tuple[str, ...]): Names of the models to request memory unload for.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    _run_memory_actions(config, model_names, "unload")


@hub.command(name="watch", help="Continuously print live hub manager status")
@click.option(
    "--interval",
    default=5.0,
    show_default=True,
    type=float,
    help="Seconds between refreshes.",
)
@click.pass_context
def hub_watch(ctx: click.Context, interval: float) -> None:
    """
    Poll the hub manager service and display live status snapshots until interrupted.

    Parameters:
        interval (float): Seconds between refreshes; values less than 0.5 are treated as 0.5.
    """
    config = _load_hub_config_or_fail(ctx.obj.get("hub_config_path"))
    click.echo("Watching hub manager (press Ctrl+C to stop)...")
    sleep_interval = max(interval, 0.5)
    try:
        while True:
            try:
                snapshot = _call_daemon_api(config, "GET", "/hub/status") or {}
            except click.ClickException:
                click.echo(
                    click.style(
                        "Hub daemon is not running. Start it via 'mlx-openai-server hub start'.",
                        fg="yellow",
                    ),
                )
            else:
                _print_watch_snapshot(snapshot)
            time.sleep(sleep_interval)
    except KeyboardInterrupt:  # pragma: no cover - interactive command
        click.echo("Stopped watching hub manager")
