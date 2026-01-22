"""Simple W&B-based model registry utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import wandb


def log_artifact(
    *,
    run: wandb.sdk.wandb_run.Run,
    checkpoint_path: Path,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_name: str,
    aliases: Optional[list[str]] = None,
) -> wandb.Artifact:
    """Log a checkpoint as a W&B model artifact."""
    meta = {"config": config, "metrics": metrics}
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=meta)
    artifact.add_file(str(checkpoint_path))
    logged = run.log_artifact(artifact, aliases=aliases or None)
    logged.wait()
    return logged


def promote_check(
    *,
    new_score: float,
    artifact_name: str,
    entity: Optional[str],
    project: str,
    metric_key: str,
) -> bool:
    """Compare new_score to current production artifact; return True if promotion is needed."""
    api = wandb.Api()
    prod_ref = (
        f"{entity}/{project}/{artifact_name}:production" if entity else f"{project}/{artifact_name}:production"
    )
    try:
        current_prod = api.artifact(prod_ref)
    except Exception:
        return True  # no production yet

    prod_score = current_prod.metadata.get("metrics", {}).get(metric_key)
    return prod_score is None or new_score > prod_score


def promote_and_demote(
    *,
    new_artifact: wandb.Artifact,
    artifact_name: str,
    entity: Optional[str],
    project: str,
) -> None:
    """Set production alias on new artifact and move old production to staging."""
    api = wandb.Api()
    prod_ref = (
        f"{entity}/{project}/{artifact_name}:production" if entity else f"{project}/{artifact_name}:production"
    )
    try:
        current_prod = api.artifact(prod_ref)
    except Exception:
        current_prod = None

    # promote new artifact
    new_aliases = list(set(list(new_artifact.aliases) + ["production"]))
    new_artifact.aliases = new_aliases
    new_artifact.save()

    if current_prod is not None:
        aliases = [a for a in current_prod.aliases if a != "production"]
        if "staging" not in aliases:
            aliases.append("staging")
        current_prod.aliases = aliases
        current_prod.save()
