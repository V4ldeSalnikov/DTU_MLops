"""Drift detection between OrganAMNIST training data and API_received uploads."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchdrift
import typer

from dtu_mlops.data import MedMNIST_dataset
from dtu_mlops.model import resnet18


class ImageFolderDataset(Dataset):
    """Lightweight image folder dataset with fixed transform."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.paths = sorted(
            [p for p in root.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), 0


class ResNetFeatureExtractor(nn.Module):
    """Forward pass without final FC layer; returns flattened features."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def _load_model(repo_id: str, filename: str, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    model = resnet18(num_classes=11, in_channels=1)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    feature_extractor = ResNetFeatureExtractor(model).to(device).eval()
    return model, feature_extractor


def _extract_normalize_transform(ds: MedMNIST_dataset) -> transforms.Normalize:
    for tr in ds.ds.transform.transforms:
        if isinstance(tr, transforms.Normalize):
            return tr
    raise RuntimeError("Normalize transform not found in MedMNIST dataset.")


def _build_reference_loader(
    data_path: Path, size: int, data_flag: str, batch_size: int, num_workers: int, max_samples: int
) -> tuple[DataLoader, transforms.Normalize, int]:
    train_ds = MedMNIST_dataset(data_path=data_path, data_flag=data_flag, split="train", data_stat=True, size=size)
    normalize = _extract_normalize_transform(train_ds)
    sample_count = min(max_samples, len(train_ds))
    subset = Subset(train_ds, list(range(sample_count)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, normalize, sample_count


def _build_production_loader(
    repo_id: str,
    local_dir: Path,
    normalize: transforms.Normalize,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, int]:
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="uploads/*",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(normalize.mean, normalize.std),
        ]
    )
    uploads_root = Path(local_path) / "uploads"
    ds = ImageFolderDataset(uploads_root, transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, len(ds)


def run_detection(
    feature_extractor: torch.nn.Module,
    ref_loader: DataLoader,
    prod_loader: DataLoader,
    alpha: float,
    device: torch.device,
) -> tuple[float, bool]:
    detector = torchdrift.detectors.KernelMMDDriftDetector(return_p_value=True)
    torchdrift.utils.fit(ref_loader, feature_extractor, detector, device=device)
    prod_outputs = _collect_outputs(prod_loader, feature_extractor, device)
    p_value = detector(prod_outputs).item()
    return p_value, p_value < alpha


def _collect_outputs(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs = []
    for batch in loader:
        if not isinstance(batch, torch.Tensor):
            batch = batch[0]
        with torch.no_grad():
            outputs.append(model(batch.to(device)))
    return torch.cat(outputs, dim=0)


def main(
    hf_model_repo: str = typer.Option("G44mlops/ResNet-medmnist", help="HF repo for the trained ResNet-18 checkpoint"),
    hf_model_filename: str = typer.Option("resnet18_best.pth", help="Checkpoint filename in the HF repo"),
    data_flag: str = typer.Option("organamnist", help="Reference MedMNIST dataset flag"),
    data_path: Path = typer.Option(Path("data/medmnist"), help="Local MedMNIST cache"),
    api_repo: str = typer.Option("G44mlops/API_received", help="HF dataset repo containing uploads to evaluate"),
    api_local_dir: Path = typer.Option(Path("data/api_received"), help="Local cache for API_received snapshots"),
    size: int = typer.Option(224, help="Image size to request from MedMNIST"),
    train_samples: int = typer.Option(10_000, help="Number of training samples to form reference distribution"),
    batch_size: int = typer.Option(128, help="Batch size for loaders"),
    num_workers: int = typer.Option(4, help="DataLoader workers"),
    alpha: float = typer.Option(0.05, help="Significance level for drift flag"),
    device: str | None = typer.Option(None, help="Device override, e.g., 'cuda' or 'cpu'"),
) -> None:
    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model, feature_extractor = _load_model(hf_model_repo, hf_model_filename, torch_device)

    ref_loader, normalize, ref_count = _build_reference_loader(
        data_path=data_path,
        size=size,
        data_flag=data_flag,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=train_samples,
    )

    prod_loader, prod_count = _build_production_loader(
        repo_id=api_repo,
        local_dir=api_local_dir,
        normalize=normalize,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    p_value, drift = run_detection(feature_extractor, ref_loader, prod_loader, alpha, torch_device)

    print(f"Reference samples: {ref_count}")
    print(f"Production samples: {prod_count}")
    print(f"p-value: {p_value:.6f}")
    print(f"Drift detected (alpha={alpha}): {drift}")


if __name__ == "__main__":
    typer.run(main)
