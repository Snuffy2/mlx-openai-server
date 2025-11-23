"""CLI regression tests for hub action subcommands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
import pytest

from app.cli import _render_watch_table, cli


@pytest.fixture
def hub_config_file(tmp_path: Path) -> Path:
    """Write a minimal hub.yaml used for CLI tests."""

    config = tmp_path / "hub.yaml"
    log_dir = tmp_path / "logs"
    config.write_text(
        f"""
log_path: {log_dir}
models:
  - name: alpha
    model_path: /models/a
    model_type: lm
""".strip()
    )
    return config


class _StubServiceClient:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reload_calls = 0
        self.shutdown_called = False

    def start_model(self, name: str) -> None:
        self.started.append(name)

    def stop_model(self, name: str) -> None:
        self.stopped.append(name)

    def reload(self) -> dict[str, list[str]]:
        self.reload_calls += 1
        return {"started": [], "stopped": [], "unchanged": []}

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_hub_reload_cli_reloads_service(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub reload` should trigger a service reload via HubServiceClient."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "reload"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1


def test_hub_stop_cli_requests_shutdown(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub stop` should reload config then shut down the service."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1
    assert stub.shutdown_called is True


def test_hub_load_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub load` should instruct the HubServiceClient to start models."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "load", "alpha"])

    assert result.exit_code == 0
    assert stub.started == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_unload_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub unload` should request stop_model for the provided names."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "unload", "alpha"])

    assert result.exit_code == 0
    assert stub.stopped == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_load_cli_requires_model_names(hub_config_file: Path) -> None:
    """The CLI should fail fast if no model names are provided."""

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "load"])

    assert result.exit_code != 0
    assert "Missing argument 'MODEL_NAMES...'" in result.output


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
