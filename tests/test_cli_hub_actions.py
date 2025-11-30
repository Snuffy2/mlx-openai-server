"""CLI regression tests for hub action subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner
import pytest

from app.cli import _render_watch_table, cli
from app.hub.config import MLXHubConfig

# `hub_config_file` is provided by `tests/conftest.py`


# `_StubServiceClient` is provided by `tests/conftest.py` as the
# `stub_service_client` fixture.


def test_hub_reload_cli_reloads_service(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub reload` should trigger a service reload via HubServiceClient."""
    stub = stub_service_client

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """
        Handle test stub calls to the daemon API for specific endpoints used by CLI tests.

        This stub responds to:
        - POST /hub/reload: delegates to the test `stub.reload()` and returns its result.
        - GET /health: returns {"status": "ok"}.

        Parameters:
            _config: Ignored configuration object included to match the real call signature.
            method: HTTP method of the request (e.g., "GET", "POST").
            path: Request path being called.
            json: Optional JSON payload (ignored).
            timeout: Request timeout in seconds (ignored).

        Returns:
            A dictionary with the endpoint response (the reload result or health status).

        Raises:
            RuntimeError: If called with any method/path combination other than the two supported above.
        """
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "reload"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1


def test_hub_stop_cli_requests_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub stop` should reload config then shut down the service."""
    stub = stub_service_client

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        # emulate availability check and reload/shutdown behavior
        """
        Test stub that simulates the daemon HTTP API and triggers side effects on the provided service stub.

        Parameters:
            _config (MLXHubConfig): Hub configuration passed by the caller (not used by the stub).
            method (str): HTTP method of the simulated request (e.g., "GET", "POST").
            path (str): Request path to route (e.g., "/health", "/hub/reload", "/hub/models/<name>/start").
            json (dict[str, object] | None): Request JSON payload if any (ignored by this stub).
            timeout (float): Request timeout in seconds (ignored by this stub).

        Returns:
            dict[str, Any]: A response-like dictionary for the handled endpoint (e.g., {"status": "ok"}, {"message": "started"}).

        Raises:
            RuntimeError: If the method/path combination is not recognized by the stub.

        Side effects:
            May call methods on the test `stub` object: `reload()`, `shutdown()`, `start_model(name)`, and `stop_model(name)`, causing observable test-side state changes.
        """
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "POST" and path == "/hub/shutdown":
            stub.shutdown()
            return {"message": "shutdown"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            name = path.split("/")[-2]
            stub.start_model(name)
            return {"message": "started"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            name = path.split("/")[-2]
            stub.stop_model(name)
            return {"message": "stopped"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)
    # Prior tests used an internal build hook; the CLI now uses the daemon API.

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1, (
        f"reloads={stub.reload_calls} availability_checks={stub.is_available_calls}"
    )
    assert stub.shutdown_called is True


def test_hub_start_model_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub start-model` should instruct the HubServiceClient to start models."""
    stub = stub_service_client

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """
        Handle stubbed daemon API calls used by CLI tests.

        Supports the following endpoints:
        - POST /hub/models/{name}/start: invokes stub.start_model(name) and returns {"message": "started"}.
        - POST /hub/reload: returns the value from stub.reload().
        - GET /health: returns {"status": "ok"}.

        Parameters:
            _config (MLXHubConfig): Ignored; present to match the real API signature.
            method (str): HTTP method of the simulated request (e.g., "GET", "POST").
            path (str): Request path; used to determine which stub action to perform.
            json (dict[str, object] | None): Ignored payload included for signature compatibility.
            timeout (float): Ignored timeout included for signature compatibility.

        Returns:
            dict[str, Any]: The simulated JSON response for the handled endpoint.

        Raises:
            RuntimeError: If an unexpected method/path combination is received.
        """
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            name = path.split("/")[-2]
            stub.start_model(name)
            return {"message": "started"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "start-model", "alpha"])

    assert result.exit_code == 0
    assert stub.started == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_stop_model_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub stop-model` should request stop_model for the provided names."""
    stub = stub_service_client

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """
        Test stub that simulates daemon API responses for a subset of hub endpoints used by CLI tests.

        Routes supported:
        - POST /hub/models/{name}/stop: records a stop on the stubbed service and returns {"message": "stopped"}.
        - POST /hub/reload: delegates to the stub's reload() and returns its result.
        - GET /health: returns {"status": "ok"}.

        Parameters:
            _config (MLXHubConfig): Ignored; present to match the real API signature.
            method (str): HTTP method of the simulated request (e.g., "GET", "POST").
            path (str): Request path; used to determine which stubbed action to invoke.
            json (dict[str, object] | None): Ignored payload for the stub.
            timeout (float): Ignored timeout value for the stub.

        Returns:
            dict[str, Any]: A JSON-like response for the handled route.

        Raises:
            RuntimeError: If the stub receives an unexpected method/path combination.
        """
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            name = path.split("/")[-2]
            stub.stop_model(name)
            return {"message": "stopped"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop-model", "alpha"])

    assert result.exit_code == 0
    assert stub.stopped == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_start_model_cli_requires_model_names(hub_config_file: Path) -> None:
    """The CLI should fail fast if no model names are provided."""
    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "start-model"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "MODEL_NAMES" in result.output


def test_render_watch_table_formats_columns() -> None:
    """The watch table helper should render sorted, columnized rows."""
    models = [
        {
            "name": "beta",
            "state": "failed",
            "pid": 2222,
            "group": "vlm",
            "started_at": 1950.0,
            "exit_code": 1,
            "log_path": "/tmp/logs/beta.log",
        },
        {
            "name": "alpha",
            "state": "running",
            "pid": 1111,
            "group": "lm",
            "started_at": 1900.0,
            "exit_code": None,
            "log_path": "/tmp/logs/alpha.log",
        },
    ]

    table = _render_watch_table(models, now=2000.0)

    assert "NAME" in table.splitlines()[0]
    assert "alpha" in table
    assert "1m40s" in table  # uptime derived from now - started_at
    assert "beta.log" in table
    assert "EXIT" in table


def test_render_watch_table_handles_empty_payload() -> None:
    """The watch table helper should gracefully render empty snapshots."""
    assert _render_watch_table([], now=0) == "  (no managed processes)"


def test_hub_load_model_cli_calls_controller(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
) -> None:
    """`hub load-model` should delegate to the controller helper."""
    captured: list[tuple[tuple[str, ...], str]] = []

    def _fake_run_actions(_config: object, names: tuple[str, ...], action: str) -> None:
        """
        Capture model action requests for test assertions.

        Appends a tuple (names, action) to the surrounding `captured` list so tests can verify which model names and action were requested.

        Parameters:
            _config (object): Configuration object (unused by this helper).
            names (tuple[str, ...]): Model names targeted by the action.
            action (str): Action to perform for the given model names (e.g., "load", "unload").
        """
        captured.append((names, action))

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hub",
            "--config",
            str(hub_config_file),
            "load-model",
            "alpha",
            "beta",
        ],
    )

    assert result.exit_code == 0
    assert captured == [(("alpha", "beta"), "load")]


def test_hub_unload_model_cli_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
) -> None:
    """`hub unload-model` should propagate helper failures as CLI errors."""

    def _fake_run_actions(_config: object, _names: tuple[str, ...], _action: str) -> None:
        """
        Always raises a ClickException with the message "boom: test".

        Raises:
            click.ClickException: Raised unconditionally to simulate a failing action.
        """
        raise click.ClickException("boom: test")

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["hub", "--config", str(hub_config_file), "unload-model", "alpha"],
    )

    assert result.exit_code != 0
    assert "boom: test" in result.output


def test_hub_config_option_loads_specified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--config` option should load the specified config file, not the default."""
    # Create two different config files
    config1 = tmp_path / "config1.yaml"
    config1.write_text(
        """
models:
  - name: model-from-config1
    model_path: /path/from/config1
    model_type: lm
""".strip()
    )

    config2 = tmp_path / "config2.yaml"
    config2.write_text(
        """
models:
  - name: model-from-config2
    model_path: /path/from/config2
    model_type: lm
""".strip()
    )

    # Mock _load_hub_config_or_fail to capture the path it was called with
    loaded_paths: list[str] = []

    def mock_load_config(config_path: str | None) -> MLXHubConfig:
        """
        Create a minimal MLXHubConfig for tests and record the provided config path.

        Parameters:
            config_path (str | None): Path to the configuration file provided to the CLI, or `None` if none was supplied.

        Returns:
            MLXHubConfig: A minimal configuration with an empty models list, host "127.0.0.1", port 8000, status page disabled, and `log_path` set to the test temporary logs directory.
        """
        loaded_paths.append(str(config_path) if config_path else "None")
        # Return a minimal config for the test
        return MLXHubConfig(
            models=[],
            host="127.0.0.1",
            port=8000,
            enable_status_page=False,
            log_path=tmp_path / "logs",
        )

    monkeypatch.setattr("app.cli._load_hub_config_or_fail", mock_load_config)

    runner = CliRunner()
    # Test with config1
    result = runner.invoke(cli, ["hub", "--config", str(config1), "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == str(config1)

    # Clear for next test
    loaded_paths.clear()

    # Test with config2
    result = runner.invoke(cli, ["hub", "--config", str(config2), "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == str(config2)

    # Clear for next test
    loaded_paths.clear()

    # Test without --config option (should use None, which means default)
    result = runner.invoke(cli, ["hub", "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == "None"  # None is passed when no --config option
