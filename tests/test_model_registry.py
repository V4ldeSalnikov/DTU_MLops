from pathlib import Path
from unittest.mock import MagicMock

import pytest

import dtu_mlops.model_registry as registry


# ============================================================
# log_artifact tests
# ============================================================
def test_log_artifact_creates_and_logs_artifact(monkeypatch, tmp_path: Path):
    """
    What we test:
    - wandb.Artifact is created with correct name/type/metadata
    - artifact.add_file() called with the checkpoint path
    - run.log_artifact() is called and wait() is executed
    - function returns the logged artifact object
    """

    ckpt = tmp_path / "model.pth"
    ckpt.write_text("dummy", encoding="utf-8")

    # Mock wandb.Artifact constructor
    artifact_obj = MagicMock()
    monkeypatch.setattr(registry.wandb, "Artifact", MagicMock(return_value=artifact_obj))

    # Mock run.log_artifact() return value (must have wait())
    logged_artifact = MagicMock()
    logged_artifact.wait.return_value = None

    run = MagicMock()
    run.log_artifact.return_value = logged_artifact

    out = registry.log_artifact(
        run=run,
        checkpoint_path=ckpt,
        config={"lr": 1e-3},
        metrics={"acc": 0.9},
        artifact_name="my_model",
        aliases=["staging"],
    )

    # Artifact constructor used correctly
    registry.wandb.Artifact.assert_called_once()
    args, kwargs = registry.wandb.Artifact.call_args
    assert kwargs["name"] == "my_model"
    assert kwargs["type"] == "model"
    assert kwargs["metadata"]["config"] == {"lr": 1e-3}
    assert kwargs["metadata"]["metrics"] == {"acc": 0.9}

    # Artifact file added
    artifact_obj.add_file.assert_called_once_with(str(ckpt))

    # Logged with aliases and waited
    run.log_artifact.assert_called_once()
    logged_artifact.wait.assert_called_once()

    assert out is logged_artifact


def test_log_artifact_aliases_none(monkeypatch, tmp_path: Path):
    """
    If aliases is None, the code passes aliases=None into run.log_artifact
    (because it uses aliases or None).
    """

    ckpt = tmp_path / "model.pth"
    ckpt.write_text("dummy", encoding="utf-8")

    artifact_obj = MagicMock()
    monkeypatch.setattr(registry.wandb, "Artifact", MagicMock(return_value=artifact_obj))

    logged_artifact = MagicMock()
    run = MagicMock()
    run.log_artifact.return_value = logged_artifact

    registry.log_artifact(
        run=run,
        checkpoint_path=ckpt,
        config={},
        metrics={},
        artifact_name="x",
        aliases=None,
    )

    # Ensure aliases=None was passed through (not [])
    _, kwargs = run.log_artifact.call_args
    assert kwargs["aliases"] is None


# ============================================================
# promote_check tests
# ============================================================
def test_promote_check_returns_true_if_no_production_exists(monkeypatch):
    """
    If api.artifact(prod_ref) raises, promote_check returns True.
    """
    api = MagicMock()
    api.artifact.side_effect = Exception("not found")

    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    assert registry.promote_check(
        new_score=0.8,
        artifact_name="mymodel",
        entity="ent",
        project="proj",
        metric_key="best_val_acc",
    )


def test_promote_check_returns_true_if_prod_has_no_metric(monkeypatch):
    """
    If production artifact exists but doesn't contain the metric key -> promote True.
    """
    current_prod = MagicMock()
    current_prod.metadata = {"metrics": {}}

    api = MagicMock()
    api.artifact.return_value = current_prod
    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    assert registry.promote_check(
        new_score=0.8,
        artifact_name="mymodel",
        entity=None,
        project="proj",
        metric_key="best_val_acc",
    )


def test_promote_check_promotes_if_new_score_higher(monkeypatch):
    """
    If new_score > current production metric => promote True
    """
    current_prod = MagicMock()
    current_prod.metadata = {"metrics": {"best_val_acc": 0.70}}

    api = MagicMock()
    api.artifact.return_value = current_prod
    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    assert registry.promote_check(
        new_score=0.80,
        artifact_name="mymodel",
        entity=None,
        project="proj",
        metric_key="best_val_acc",
    )


def test_promote_check_does_not_promote_if_new_score_lower(monkeypatch):
    """
    If new_score <= current production metric => promote False
    """
    current_prod = MagicMock()
    current_prod.metadata = {"metrics": {"best_val_acc": 0.90}}

    api = MagicMock()
    api.artifact.return_value = current_prod
    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    assert not registry.promote_check(
        new_score=0.80,
        artifact_name="mymodel",
        entity="ent",
        project="proj",
        metric_key="best_val_acc",
    )


# ============================================================
# promote_and_demote tests
# ============================================================
def test_promote_and_demote_when_no_current_prod(monkeypatch):
    """
    If no current production exists:
    - new artifact gets 'production' alias and is saved
    - no demotion is performed
    """
    api = MagicMock()
    api.artifact.side_effect = Exception("no prod")
    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    new_artifact = MagicMock()
    new_artifact.aliases = ["staging"]

    registry.promote_and_demote(
        new_artifact=new_artifact,
        artifact_name="mymodel",
        entity=None,
        project="proj",
    )

    assert "production" in new_artifact.aliases
    new_artifact.save.assert_called_once()


def test_promote_and_demote_demotes_old_prod_to_staging(monkeypatch):
    """
    If current production exists:
    - new artifact gets 'production'
    - old production loses 'production' and gains 'staging'
    """
    current_prod = MagicMock()
    current_prod.aliases = ["production"]

    api = MagicMock()
    api.artifact.return_value = current_prod
    monkeypatch.setattr(registry.wandb, "Api", MagicMock(return_value=api))

    new_artifact = MagicMock()
    new_artifact.aliases = ["staging"]

    registry.promote_and_demote(
        new_artifact=new_artifact,
        artifact_name="mymodel",
        entity="ent",
        project="proj",
    )

    # New promoted
    assert "production" in new_artifact.aliases
    new_artifact.save.assert_called_once()

    # Old demoted
    assert "production" not in current_prod.aliases
    assert "staging" in current_prod.aliases
    current_prod.save.assert_called_once()
