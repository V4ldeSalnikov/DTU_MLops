"""Evaluate a checkpoint from the Hugging Face Hub on the MedMNIST val split."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
import typer

from dtu_mlops.data import MedMNIST_dataset
from dtu_mlops.model import resnet18, resnet50


def _load_model(
    checkpoint_path: Path,
    model_type: str,
    num_classes: int,
    in_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate model and load checkpoint."""
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_type == "resnet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError("model_type must be 'resnet18' or 'resnet50'")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy."""
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(-1).long()
            logits = model(images)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total if total else 0.0


def main(
    repo_id: str = typer.Option(
        "V4ldeLund/ResNet-medmnist", help="HF repo id containing the checkpoint"
    ),
    filename: str = typer.Option(
        "resnet18_best.pth", help="File name inside the HF repo (e.g., pytorch_model.bin)"
    ),
    model_type: str = typer.Option("resnet18", help="resnet18 or resnet50"),
    data_flag: str = typer.Option("organamnist", help="MedMNIST dataset flag"),
    data_path: Path = typer.Option(Path("data/medmnist"), help="Local cache for MedMNIST"),
    batch_size: int = typer.Option(64, help="Validation batch size"),
    num_workers: int = typer.Option(4, help="Data loader workers"),
    num_classes: int = typer.Option(11, help="Number of classes"),
    in_channels: int = typer.Option(1, help="Input channels (1 for grayscale)"),
    size: int = typer.Option(224, help="Image size used during training"),
) -> None:
    """Download a checkpoint from HF and report validation accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
    )

    model = _load_model(
        checkpoint_path=ckpt_path,
        model_type=model_type,
        num_classes=num_classes,
        in_channels=in_channels,
        device=device,
    )

    val_ds = MedMNIST_dataset(
        data_path=data_path,
        data_flag=data_flag,
        split="val",
        data_stat=True,
        size=size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    acc = _accuracy(model, val_loader, device)
    print(f"Validation accuracy: {acc:.2f}% on {len(val_ds)} samples (device={device})")


if __name__ == "__main__":
    typer.run(main)
