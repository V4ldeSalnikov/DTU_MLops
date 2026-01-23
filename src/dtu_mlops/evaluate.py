import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Any, Dict, Optional, cast
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dtu_mlops.config_utils import resolve_param, validate_required_keys
from dtu_mlops.data import MedMNIST_dataset
from dtu_mlops.model import resnet18, resnet50

import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hydra_utils

from medmnist import INFO


def load_model_from_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    num_classes: int,
    in_channels: int,
    device: torch.device,
) -> nn.Module:
    """
    Rebuilds the model architecture and loads the saved weights.

    We need to know the architecture (e.g. ResNet18) and input/output shapes
    to construct the model correctly before populating it with the trained weights.

    Returns:
        The model, ready for evaluation (in eval mode and on the correct device).
    """
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_type == "resnet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose 'resnet18' or 'resnet50'"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the model on the entire dataset to gather predictions.

    Returns:
        A tuple containing:
        - The predicted class indices
        - The actual ground truth labels
        - The raw probabilities for each class
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(-1).long()

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """
    Calculates common classification metrics like Accuracy, Precision, Recall, and F1.

    We compute these in different ways (macro vs weighted) to handle class imbalance better.
    We also compute per-class metrics to see if the model struggles with any specific class.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    precision_per_class = np.array(
        precision_score(y_true, y_pred, average=None, zero_division=0)
    )
    recall_per_class = np.array(
        recall_score(y_true, y_pred, average=None, zero_division=0)
    )
    f1_per_class = np.array(f1_score(y_true, y_pred, average=None, zero_division=0))

    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = precision_per_class[i]
        metrics[f"recall_class_{i}"] = recall_per_class[i]
        metrics[f"f1_class_{i}"] = f1_per_class[i]

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Creates a visual confusion matrix to see where the model is making mistakes.

    If 'normalize' is True, it shows percentages instead of raw counts, which is
    usually easier to read for imbalanced datasets.
    """
    conf_mat = confusion_matrix(y_true, y_pred)

    if normalize:
        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1, keepdims=True)
        conf_mat = np.nan_to_num(conf_mat)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_mat.shape[1]),
        yticks=np.arange(conf_mat.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Normalized Confusion Matrix" if normalize else "Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], fmt),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def evaluate(
    cfg: DictConfig,
    checkpoint_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    data_flag: Optional[str] = None,
    model_type: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    split: str = "test",
    limit_samples: Optional[int] = None,
) -> dict[str, float]:
    """
    Main function for evaluation.

    1. Sets up the device and output directories.
    2. Loads the test data and the trained model.
    3. Runs inference.
    4. Calculates metrics and saves reports/plots.
    5. Logs everything to WandB.
    """
    eval_cfg = cast(Dict[str, Any], cfg)
    validate_required_keys(
        eval_cfg,
        [
            "data_path",
            "data_flag",
            "batch_size",
            "num_workers",
            "device",
        ],
    )

    data_path = cast(
        Path, resolve_param(data_path, eval_cfg, "data_path", as_path=True)
    )
    data_flag = cast(str, resolve_param(data_flag, eval_cfg, "data_flag"))
    batch_size = cast(int, resolve_param(batch_size, eval_cfg, "batch_size"))
    limit_samples = (
        resolve_param(limit_samples, eval_cfg, "limit_samples")
        if limit_samples is not None or "limit_samples" in eval_cfg
        else None
    )
    num_workers = cast(int, resolve_param(num_workers, eval_cfg, "num_workers"))
    device_str = resolve_param(device, eval_cfg, "device")

    checkpoint_path = checkpoint_path or eval_cfg.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided either via CLI or config")
    checkpoint_path = Path(checkpoint_path)

    output_dir = output_dir or eval_cfg.get("output_dir")
    if output_dir is None:
        output_dir = Path("reports/evaluation")
    else:
        output_dir = Path(output_dir)

    model_cfg = eval_cfg.get("model") if "model" in eval_cfg else None
    resolved_model_type = model_type
    if resolved_model_type is None:
        if model_cfg and model_cfg.get("model_type") is not None:
            resolved_model_type = model_cfg.model_type
        elif eval_cfg.get("model_type"):
            resolved_model_type = eval_cfg.get("model_type")
    if resolved_model_type is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        resolved_model_type = checkpoint.get("model_type", "resnet18")
    model_type = cast(str, resolved_model_type)

    num_classes = (
        model_cfg.get("num_classes")
        if model_cfg and model_cfg.get("num_classes") is not None
        else eval_cfg.get("num_classes", 11)
    )
    in_channels = (
        model_cfg.get("in_channels")
        if model_cfg and model_cfg.get("in_channels") is not None
        else eval_cfg.get("in_channels", 1)
    )

    info = INFO[data_flag]
    class_names = info.get("label", {})
    if isinstance(class_names, dict):
        class_names = [
            class_names.get(str(i), f"Class {i}") for i in range(num_classes)
        ]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    print("=" * 60)
    print("Evaluation Configuration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset: {data_flag}")
    print(f"  Split: {split}")
    print(f"  Model: {model_type}")
    if limit_samples:
        print(f"  Sample limit: {limit_samples}")
    print(f"  Batch size: {batch_size}")
    print("=" * 60)

    if device_str is None:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        eval_device = torch.device(device_str)
    print(f"Using device: {eval_device}")

    wandb_entity = eval_cfg.get(
        "wandb_entity", "v4lde-danmarks-tekniske-universitet-dtu"
    )
    wandb_project = eval_cfg.get("wandb_project", "DTU-MLops")
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=eval_cfg.get("wandb_mode", "online"),
        name=eval_cfg.get("wandb_name", f"eval-{model_type}-{split}"),
        tags=eval_cfg.get("wandb_tags", ["evaluation"]),
        job_type="evaluation",
        config={
            "checkpoint_path": str(checkpoint_path),
            "data_path": str(data_path),
            "data_flag": data_flag,
            "split": split,
            "model_type": model_type,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "device": str(eval_device),
            "_hydra_config": OmegaConf.to_container(eval_cfg, resolve=True),
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset...")
    eval_dataset = MedMNIST_dataset(
        data_path=data_path,
        data_flag=data_flag,
        split=split,
        data_stat=True,
    )

    if limit_samples is not None:
        full_size = len(eval_dataset)
        if limit_samples > full_size:
            print(
                f"Warning: limit_samples ({limit_samples}) > dataset size ({full_size}). Using full dataset."
            )
        else:
            indices = torch.randperm(full_size)[:limit_samples]
            eval_dataset = Subset(eval_dataset, indices.tolist())
            print(f"Using subset of {limit_samples} samples")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if eval_device.type == "cuda" else False,
    )

    print(f"Evaluation samples: {len(eval_dataset)}")

    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        num_classes=num_classes,
        in_channels=in_channels,
        device=eval_device,
    )

    print("\nRunning evaluation...")
    y_pred, y_true, y_probs = get_predictions(model, eval_loader, eval_device)

    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, num_classes)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    print("=" * 60)

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    report_path = output_dir / f"classification_report_{split}.txt"
    with open(report_path, "w") as f:
        f.write(f"Evaluation Results for {checkpoint_path}\n")
        f.write(f"Dataset: {data_flag}, Split: {split}\n")
        f.write("=" * 60 + "\n\n")
        if isinstance(report, str):
            f.write(report)
        else:
            f.write(str(report))
    print(f"Classification report saved to: {report_path}")

    print("\nGenerating confusion matrix...")
    cm_path = output_dir / f"confusion_matrix_{split}.png"
    fig = plot_confusion_matrix(
        y_true, y_pred, class_names, normalize=True, save_path=cm_path
    )

    wandb.log(
        {
            f"eval/{split}/accuracy": metrics["accuracy"],
            f"eval/{split}/precision_macro": metrics["precision_macro"],
            f"eval/{split}/precision_weighted": metrics["precision_weighted"],
            f"eval/{split}/recall_macro": metrics["recall_macro"],
            f"eval/{split}/recall_weighted": metrics["recall_weighted"],
            f"eval/{split}/f1_macro": metrics["f1_macro"],
            f"eval/{split}/f1_weighted": metrics["f1_weighted"],
        }
    )

    for i in range(num_classes):
        wandb.log(
            {
                f"eval/{split}/precision_class_{i}": metrics[f"precision_class_{i}"],
                f"eval/{split}/recall_class_{i}": metrics[f"recall_class_{i}"],
                f"eval/{split}/f1_class_{i}": metrics[f"f1_class_{i}"],
            }
        )

    wandb.log({f"eval/{split}/confusion_matrix": wandb.Image(fig)})

    wandb_run.summary[f"{split}_accuracy"] = metrics["accuracy"]
    wandb_run.summary[f"{split}_f1_macro"] = metrics["f1_macro"]
    wandb_run.summary[f"{split}_precision_macro"] = metrics["precision_macro"]
    wandb_run.summary[f"{split}_recall_macro"] = metrics["recall_macro"]

    plt.close(fig)

    wandb_run.finish()

    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return metrics


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point for the script.
    It resolves paths to be absolute (since Hydra changes the working directory)
    and then starts the evaluation.
    """
    orig_cwd = Path(hydra_utils.get_original_cwd())

    if cfg.get("data_path"):
        cfg.data_path = str((orig_cwd / cfg.data_path).resolve())
    if cfg.get("checkpoint_path"):
        cfg.checkpoint_path = str((orig_cwd / cfg.checkpoint_path).resolve())
    if cfg.get("output_dir"):
        cfg.output_dir = str((orig_cwd / cfg.output_dir).resolve())

    evaluate(
        cfg=cfg,
        split=cfg.get("split", "test"),
    )


if __name__ == "__main__":
    main()
