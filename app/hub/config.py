"""Hub configuration parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from loguru import logger
import yaml

from ..config import MLXServerConfig

HUB_CONFIG_FILENAME = "hub.yaml"
HUB_HOME = Path("~/mlx-openai-server").expanduser()
DEFAULT_HUB_CONFIG_PATH = HUB_HOME / HUB_CONFIG_FILENAME
DEFAULT_HUB_LOG_PATH = HUB_HOME / "logs"


class HubConfigError(RuntimeError):
    """Raised when the hub configuration file is invalid."""


_slug_pattern = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$", re.IGNORECASE)


def _ensure_slug(value: str, *, field_name: str) -> str:
    """Validate that ``value`` is already a compliant slug without altering it."""

    candidate = value.strip()
    if not candidate:
        raise HubConfigError(f"{field_name} cannot be empty")
    if not _slug_pattern.fullmatch(candidate):
        raise HubConfigError(
            f"{field_name} must be alphanumeric with optional hyphen/underscore separators"
        )
    return candidate


@dataclass(slots=True)
class MLXHubGroupConfig:
    """Configuration for a logical model group."""

    name: str
    max_loaded: int | None = None

    def __post_init__(self) -> None:
        """Validate group names and ensure ``max_loaded`` when provided is sane."""
        self.name = _ensure_slug(self.name, field_name="group name")
        if self.max_loaded is not None and self.max_loaded < 1:
            raise HubConfigError("max_loaded must be a positive integer when provided")


@dataclass(slots=True)
class MLXHubConfig:
    """Top-level hub configuration derived from YAML."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_path: Path = field(default_factory=lambda: DEFAULT_HUB_LOG_PATH)
    enable_status_page: bool = True
    models: list[MLXServerConfig] = field(default_factory=list)
    groups: list[MLXHubGroupConfig] = field(default_factory=list)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalize hub defaults and ensure log directories exist."""
        self.log_level = self.log_level.upper()
        self.log_path = self.log_path.expanduser()
        self.log_path.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        The parsed YAML data.

    Raises
    ------
    HubConfigError
        If the file is not found or parsing fails.
    """

    try:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise HubConfigError(f"Hub config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise HubConfigError(f"Failed to parse hub config '{path}': {exc}") from exc

    if not isinstance(loaded, dict):
        raise HubConfigError("Hub config root must be a mapping")
    return loaded


def _build_groups(raw_groups: list[dict[str, Any]] | None) -> list[MLXHubGroupConfig]:
    """Build group configurations from raw data.

    Parameters
    ----------
    raw_groups : list[dict[str, Any]] or None
        Raw group data from YAML.

    Returns
    -------
    list[MLXHubGroupConfig]
        List of group configurations.

    Raises
    ------
    HubConfigError
        If group data is invalid.
    """

    groups: list[MLXHubGroupConfig] = []
    if not raw_groups:
        return groups

    seen: set[str] = set()
    for idx, group_data in enumerate(raw_groups, start=1):
        if not isinstance(group_data, dict):
            raise HubConfigError(f"Group entry #{idx} must be a mapping")
        if "name" not in group_data:
            raise HubConfigError(f"Group entry #{idx} is missing required 'name'")
        max_loaded_value = group_data.get("max_loaded")
        max_loaded = None
        if max_loaded_value is not None:
            try:
                max_loaded = int(max_loaded_value)
            except (TypeError, ValueError) as exc:
                raise HubConfigError("max_loaded must be an integer") from exc

        group = MLXHubGroupConfig(
            name=str(group_data["name"]),
            max_loaded=max_loaded,
        )
        if group.name in seen:
            raise HubConfigError(f"Duplicate group name '{group.name}' detected")
        seen.add(group.name)
        groups.append(group)
    return groups


def _validate_group_defaults(
    models: list[MLXServerConfig], group_lookup: dict[str, MLXHubGroupConfig]
) -> None:
    """Ensure grouped default models do not exceed their caps.

    Parameters
    ----------
    models : list[MLXServerConfig]
        List of model configurations.
    group_lookup : dict[str, MLXHubGroupConfig]
        Lookup of group configurations.

    Raises
    ------
    HubConfigError
        If validation fails.
    """

    if not group_lookup:
        return

    default_counts: dict[str, int] = {}
    for name in group_lookup:
        default_counts[name] = 0
    for model in models:
        if not model.group:
            continue
        if model.group not in group_lookup:
            raise HubConfigError(f"Model '{model.name}' references undefined group '{model.group}'")
        if model.is_default_model:
            default_counts[model.group] += 1

    for group_name, count in default_counts.items():
        group_cfg = group_lookup[group_name]
        if group_cfg.max_loaded is None:
            continue
        if count > group_cfg.max_loaded:
            raise HubConfigError(
                f"Group '{group_name}' has {count} default model(s) but max_loaded is {group_cfg.max_loaded}"
            )


def _resolve_model_log_file(server_config: MLXServerConfig, hub_log_path: Path) -> MLXServerConfig:
    """Resolve the log file path for a model configuration.

    Parameters
    ----------
    server_config : MLXServerConfig
        The server configuration.
    hub_log_path : Path
        The hub log directory path.

    Returns
    -------
    MLXServerConfig
        The updated server configuration.

    Raises
    ------
    HubConfigError
        If the model name is missing.
    """

    if server_config.no_log_file:
        return server_config

    if server_config.log_file:
        return server_config

    if not server_config.name:
        raise HubConfigError("Each hub model requires a 'name' to derive default log paths")

    log_file = hub_log_path / f"{server_config.name}.log"
    server_config.log_file = str(log_file)
    return server_config


def _build_models(
    raw_models: list[dict[str, Any]] | None,
    base_host: str,
    base_port: int,
    base_log_level: str,
    hub_log_path: Path,
    group_lookup: dict[str, MLXHubGroupConfig],
) -> list[MLXServerConfig]:
    """Build model configurations from raw data.

    Parameters
    ----------
    raw_models : list[dict[str, Any]] or None
        Raw model data from YAML.
    base_host : str
        Base host for models.
    base_port : int
        Base port for models.
    base_log_level : str
        Base log level for models.
    hub_log_path : Path
        Hub log directory path.
    group_lookup : dict[str, MLXHubGroupConfig]
        Lookup of group configurations.

    Returns
    -------
    list[MLXServerConfig]
        List of model configurations.

    Raises
    ------
    HubConfigError
        If model data is invalid.
    """

    if not raw_models:
        raise HubConfigError("Hub config must include at least one model entry")

    models: list[MLXServerConfig] = []
    seen_names: set[str] = set()

    for idx, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise HubConfigError(f"Model entry #{idx} must be a mapping")

        if "model_path" not in raw_model:
            raise HubConfigError(f"Model entry #{idx} is missing required 'model_path'")
        if "name" not in raw_model:
            raise HubConfigError(f"Model entry #{idx} is missing required 'name'")

        name = _ensure_slug(str(raw_model["name"]), field_name="model name")
        group_value = raw_model.get("group")
        group_slug = _ensure_slug(str(group_value), field_name="group") if group_value else None
        if group_slug and group_lookup and group_slug not in group_lookup:
            raise HubConfigError(
                f"Model '{name}' references group '{group_slug}' which is not defined in hub config"
            )

        model_payload = dict(raw_model)
        default_flag = bool(model_payload.pop("default", False))

        model_payload["host"] = model_payload.get("host", base_host)
        model_payload["port"] = model_payload.get("port", base_port)
        model_payload["log_level"] = model_payload.get("log_level", base_log_level)
        model_payload["name"] = name
        model_payload["group"] = group_slug

        server_config = MLXServerConfig(**model_payload)
        server_config.is_default_model = default_flag
        if not server_config.name:
            raise HubConfigError("Each hub model must have a slug-compliant name")
        if server_config.name in seen_names:
            raise HubConfigError(f"Duplicate model name '{server_config.name}' detected")
        seen_names.add(server_config.name)
        server_config = _resolve_model_log_file(server_config, hub_log_path)
        models.append(server_config)

    return models


def load_hub_config(config_path: Path | str | None = None) -> MLXHubConfig:
    """Load and validate a hub configuration file.

    Parameters
    ----------
    config_path : Path, str, or None, optional
        Path to the hub configuration file. If None, uses default path.

    Returns
    -------
    MLXHubConfig
        The loaded and validated hub configuration.
    """

    if config_path is None:
        path = DEFAULT_HUB_CONFIG_PATH
    else:
        path = Path(config_path).expanduser()

    data = _load_yaml(path)

    port_value = data.get("port", 8000)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise HubConfigError("Hub port must be an integer") from exc

    hub = MLXHubConfig(
        host=str(data.get("host", "0.0.0.0")),
        port=port,
        log_level=str(data.get("log_level", "INFO")),
        log_path=Path(str(data.get("log_path", DEFAULT_HUB_LOG_PATH))),
        enable_status_page=data.get("enable_status_page", True),
        source_path=path,
    )

    hub.groups = _build_groups(data.get("groups"))
    group_lookup = {group.name: group for group in hub.groups}
    hub.models = _build_models(
        raw_models=data.get("models"),
        base_host=hub.host,
        base_port=hub.port,
        base_log_level=hub.log_level,
        hub_log_path=hub.log_path,
        group_lookup=group_lookup,
    )

    # Ensure all models inherit the hub's status page setting
    for model in hub.models:
        model.enable_status_page = hub.enable_status_page

    _validate_group_defaults(hub.models, group_lookup)

    logger.info(
        f"Loaded hub config from {path} with {len(hub.models)} model(s) and {len(hub.groups)} group(s)"
    )
    return hub
