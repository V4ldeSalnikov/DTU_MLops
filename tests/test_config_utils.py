import pytest
from pathlib import Path

from dtu_mlops.config_utils import load_yaml_config, resolve_param, validate_required_keys


# -----------------------------
# load_yaml_config tests
# -----------------------------
def test_missing_yaml_config_returns_empty_dict(tmp_path: Path):
    """
    If the yaml file doesn't exist, load_yaml_config should return {}.
    This covers the branch:
        if not path.exists(): return {}
    """
    missing_file = tmp_path / "missing.yaml"
    cfg = load_yaml_config(missing_file)
    assert cfg == {}


def test_read_yaml_config_when_file_exists(tmp_path: Path):
    """
    If the file exists and contains yaml, it should load into a dict.
    This covers the file open + yaml.safe_load branch.
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("epochs: 10\nlearning_rate: 0.001\n", encoding="utf-8")

    cfg = load_yaml_config(cfg_file)
    assert cfg["epochs"] == 10
    assert cfg["learning_rate"] == 0.001


def test_empty_yaml_config_returns_empty(tmp_path: Path):
    """
    yaml.safe_load can return None for an empty file.
    Your function does: yaml.safe_load(...) or {}
    So empty file -> {}.
    """
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("", encoding="utf-8")

    cfg = load_yaml_config(cfg_file)
    assert cfg == {}


# -----------------------------
# resolve_param tests
# -----------------------------
def test_resolve_param_prefers_cli_value_over_cfg():
    """
    The priority rule is:
        CLI > config
    So if cli_value is provided, it should return cli_value.
    """
    cfg = {"epochs": 10}
    assert resolve_param(5, cfg, "epochs") == 5


def test_resolve_param_falls_back_to_cfg_if_cli_none():
    """
    If cli_value is None, the function should return cfg[key] when present.
    """
    cfg = {"epochs": 10}
    assert resolve_param(None, cfg, "epochs") == 10


def test_resolve_param_raises_keyerror_if_missing_in_both():
    """
    If neither cli_value nor cfg provides the key, KeyError should be raised.
    """
    cfg = {}
    with pytest.raises(KeyError):
        resolve_param(None, cfg, "epochs")


def test_resolve_param_as_path_returns_path_object(tmp_path: Path):
    """
    If as_path=True, resolve_param should convert string paths into Path objects.
    Test both CLI and cfg variants (this test checks CLI variant).
    """
    cfg = {"data_path": "should_not_be_used"}
    resolved = resolve_param(str(tmp_path), cfg, "data_path", as_path=True)

    assert isinstance(resolved, Path)
    assert resolved == tmp_path


def test_resolve_param_as_path_from_cfg(tmp_path: Path):
    """
    Same as above, but if cli_value is None, it should read from cfg and convert to Path.
    """
    cfg = {"data_path": str(tmp_path)}
    resolved = resolve_param(None, cfg, "data_path", as_path=True)

    assert isinstance(resolved, Path)
    assert resolved == tmp_path


# -----------------------------
# validate_required_keys tests
# -----------------------------
def test_validate_required_keys_passes_when_all_keys_present():
    """
    If all required keys exist in cfg, validate_required_keys should not raise.
    """
    cfg = {"a": 1, "b": 2}
    validate_required_keys(cfg, ["a", "b"])


def test_validate_required_keys_raises_keyError_when_missing():
    """
    If any required key is missing, validate_required_keys should raise KeyError.
    """
    cfg = {"a": 1}
    with pytest.raises(KeyError) as excinfo:
        validate_required_keys(cfg, ["a", "b"])

    # check error message contains missing key
    assert "b" in str(excinfo.value)
