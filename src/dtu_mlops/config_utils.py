from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file if it exists, else return an empty dict."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_param(
    cli_value: Any,
    cfg: Dict[str, Any],
    key: str,
    *,
    as_path: bool = False,
) -> Any:
    """Resolve parameter priority: CLI > config file. Raises if missing."""
    if cli_value is not None:
        return Path(cli_value) if as_path else cli_value
    if key in cfg:
        return Path(cfg[key]) if as_path else cfg[key]
    raise KeyError(f"Missing required config key: {key}")


def validate_required_keys(cfg: Dict[str, Any], required: list[str]) -> None:
    """Ensure required keys are present; raise KeyError if any are missing."""
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")
