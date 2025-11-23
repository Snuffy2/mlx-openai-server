"""Tests for hub configuration parsing helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from app.hub.config import HubConfigError, MLXHubConfig, load_hub_config


def _write_yaml(path: Path, contents: str) -> Path:
    path.write_text(dedent(contents).strip(), encoding="utf-8")
    return path


def test_load_hub_config_creates_log_directory(tmp_path: Path) -> None:
    """Loading a hub config should expand and create the log directory."""

    log_dir = tmp_path / "logs" / "nested"
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        f"""
        log_path: {log_dir}
        models:
          - name: alpha
            model_path: /models/alpha
        """,
    )

    config = load_hub_config(hub_path)

    assert config.log_path == log_dir
    assert log_dir.exists() and log_dir.is_dir()


def test_load_hub_config_rejects_invalid_model_slug(tmp_path: Path) -> None:
    """Model names must already be slug-compliant."""

    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: "invalid slug"
            model_path: /models/alpha
        """,
    )

    with pytest.raises(HubConfigError, match="model name"):
        load_hub_config(hub_path)


def test_load_hub_config_allows_group_reference_when_groups_missing(tmp_path: Path) -> None:
    """Referencing a group is permitted even when no group objects are defined."""

    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
            group: workers
        """,
    )

    config = load_hub_config(hub_path)

    assert isinstance(config, MLXHubConfig)
    assert config.models[0].group == "workers"
