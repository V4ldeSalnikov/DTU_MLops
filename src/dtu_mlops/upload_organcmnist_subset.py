"""Create and upload an OrganCMNIST subset to a Hugging Face dataset.

The script samples images from the OrganCMNIST training split, saves them as PNGs
along with metadata files mirroring the InfiniteLobster/MLOps_dataset layout, and
pushes the ``uploads`` folder to the target Hugging Face dataset repository.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import medmnist
from huggingface_hub import HfApi
from medmnist import INFO
from PIL import Image
import typer


def _label_lookup() -> Dict[int, str]:
    """Return class index to name mapping for OrganCMNIST."""

    raw_map = INFO["organcmnist"]["label"]
    return {int(idx): name for idx, name in raw_map.items()}


def _load_organcmnist_train(data_root: Path, size: int) -> medmnist.dataset.MedMNIST:
    """Load the OrganCMNIST training split."""

    data_root.mkdir(parents=True, exist_ok=True)
    data_class = getattr(medmnist, INFO["organcmnist"]["python_class"])
    return data_class(root=str(data_root), split="train", download=True, size=size)


def _ensure_pil(image) -> Image.Image:
    """Convert a MedMNIST sample to ``PIL.Image`` if needed."""

    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)


def _label_to_int(label) -> int:
    """Convert MedMNIST label tensor/array to int."""

    if hasattr(label, "__len__") and not isinstance(label, (str, bytes)):
        return int(label[0])
    return int(label)


def _write_metadata(meta_path: Path, prediction: str, timestamp: str) -> None:
    """Persist metadata alongside an uploaded image."""

    content = f"prediction: {prediction}\n" f"timestamp: {timestamp}\n"
    meta_path.write_text(content)


def _stage_samples(
    dataset: medmnist.dataset.MedMNIST,
    label_map: Dict[int, str],
    uploads_dir: Path,
    indices: Iterable[int],
    run_id: str,
) -> int:
    """Save selected samples and metadata to ``uploads_dir``.

    Returns the number of staged samples.
    """

    uploads_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices, start=1):
        image, label = dataset[idx]
        pil_img = _ensure_pil(image)
        label_int = _label_to_int(label)
        class_name = label_map.get(label_int, str(label_int))
        stem = f"{run_id}_i{i}"
        img_path = uploads_dir / f"{stem}.png"
        meta_path = uploads_dir / f"{stem}_metadata.txt"
        pil_img.save(img_path)
        _write_metadata(meta_path, class_name, run_id)
    return i if "i" in locals() else 0


def _push_folder(repo_id: str, token: str, folder: Path, commit_message: str) -> None:
    """Upload a local folder to a Hugging Face dataset repository."""

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=".",
        commit_message=commit_message,
        token=token,
    )


def main(
    hf_repo_id: str = typer.Option(
        "G44mlops/API_received",
        help="Target Hugging Face dataset repository id",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        envvar="HF_TOKEN",
        help="Hugging Face access token with write permissions to the dataset",
    ),
    data_path: Path = typer.Option(
        Path("data/medmnist"),
        help="Local cache directory for MedMNIST datasets",
    ),
    staging_dir: Path = typer.Option(
        Path("outputs/organcmnist_api_payload"),
        help="Local folder where the uploads/ contents are staged",
    ),
    num_samples: int = typer.Option(1000, help="Number of OrganCMNIST train images to upload"),
    size: int = typer.Option(224, help="Requested image size from MedMNIST"),
    seed: int = typer.Option(44, help="Seed for deterministic sampling"),
    skip_upload: bool = typer.Option(False, help="If set, only stage files locally without pushing to Hugging Face"),
) -> None:
    """Stage and upload OrganCMNIST samples following the expected API format."""

    if hf_token is None and not skip_upload:
        raise typer.BadParameter("Provide HF token via --hf-token or HF_TOKEN env var when uploading.")

    dataset = _load_organcmnist_train(data_path, size)
    if num_samples > len(dataset):
        raise typer.BadParameter(f"num_samples ({num_samples}) exceeds dataset size ({len(dataset)}).")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), num_samples)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uploads_dir = staging_dir / "uploads"

    label_map = _label_lookup()
    staged = _stage_samples(dataset, label_map, uploads_dir, indices, run_id)

    if skip_upload:
        print(f"Staged {staged} samples under {uploads_dir}. Upload skipped.")
        return

    commit_message = f"Add {staged} OrganCMNIST samples ({run_id})"
    _push_folder(hf_repo_id, hf_token, staging_dir, commit_message)
    print(f"Uploaded {staged} samples to {hf_repo_id} (commit: {commit_message}).")


if __name__ == "__main__":
    typer.run(main)
