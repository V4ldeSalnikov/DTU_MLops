import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import typer
from tqdm import tqdm
import wandb

from dtu_mlops.config_utils import load_yaml_config, resolve_param, validate_required_keys
from dtu_mlops.data import MedMNIST_dataset
from dtu_mlops.model import resnet18, resnet50


app = typer.Typer()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train model for one epoch

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze(-1).long()  # MedMNIST labels have shape (batch_size, 1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(-1).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@app.command()
def train(
    train_config_path: Path = Path("configs/train.yaml"),
    data_path: Optional[Path] = None,
    data_flag: Optional[str] = None,
    model_type: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    save_best: Optional[bool] = None,
) -> None:
    """Train ResNet model on MedMNIST dataset

    Args:
        train_config_path: YAML config file with training hyperparameters
        data_path: Override for data directory (falls back to config/default)
        data_flag: Override MedMNIST dataset flag (e.g., "organamnist")
        model_type: Override model architecture ("resnet18" or "resnet50")
        epochs: Override number of training epochs
        batch_size: Override batch size
        learning_rate: Override learning rate
        weight_decay: Override weight decay
        num_workers: Override number of workers for data loading
        device: Override device to train on (None for auto-detect)
        checkpoint_dir: Override directory to save model checkpoints
        save_best: Override whether to save best model based on validation accuracy
    """
    # Load training config then resolve hyperparameters (CLI ovrwrite config)
    train_cfg = load_yaml_config(train_config_path)
    validate_required_keys(
        train_cfg,
        [
            "data_path",
            "data_flag",
            "model_type",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "num_workers",
            "device",
            "checkpoint_dir",
            "save_best",
        ],
    )
    data_path = resolve_param(data_path, train_cfg, "data_path", as_path=True)
    data_flag = resolve_param(data_flag, train_cfg, "data_flag")
    model_type = resolve_param(model_type, train_cfg, "model_type")
    epochs = resolve_param(epochs, train_cfg, "epochs")
    batch_size = resolve_param(batch_size, train_cfg, "batch_size")
    learning_rate = resolve_param(learning_rate, train_cfg, "learning_rate")
    weight_decay = resolve_param(weight_decay, train_cfg, "weight_decay")
    num_workers = resolve_param(num_workers, train_cfg, "num_workers")
    device = resolve_param(device, train_cfg, "device")
    checkpoint_dir = resolve_param(checkpoint_dir, train_cfg, "checkpoint_dir", as_path=True)
    save_best = resolve_param(save_best, train_cfg, "save_best")

    print("=" * 60)
    print("Training Configuration:")
    print(f"  Dataset: {data_flag}")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print("=" * 60)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # W&B settings
    wandb_entity = "v4lde-danmarks-tekniske-universitet-dtu"
    wandb_project = "DTU-MLops"
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode="online",
        name=train_cfg.get("wandb_name"),
        tags=train_cfg.get("wandb_tags", []),
        group=train_cfg.get("wandb_group") or None,
        notes=train_cfg.get("wandb_notes") or None,
        job_type=train_cfg.get("wandb_job_type") or None,
        config={
            "data_path": str(data_path),
            "data_flag": data_flag,
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_workers": num_workers,
            "device": str(device) if device else None,
            "checkpoint_dir": str(checkpoint_dir),
            "save_best": save_best,
        },
    )

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MedMNIST_dataset(
        data_path=data_path,
        data_flag=data_flag,
        split="train",
        data_stat=True,
    )

    val_dataset = MedMNIST_dataset(
        data_path=data_path,
        data_flag=data_flag,
        split="val",
        data_stat=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == "resnet18":
        model = resnet18(num_classes=11, in_channels=1)
    elif model_type == "resnet50":
        model = resnet50(num_classes=11, in_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'resnet18' or 'resnet50'")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True,
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step(val_acc)

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_type': model_type,
            'data_flag': data_flag,
        }

        # Save last checkpoint
        torch.save(checkpoint, checkpoint_dir / f"{model_type}_last.pth")

        # Save best checkpoint
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, checkpoint_dir / f"{model_type}_best.pth")
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            wandb_run.summary["best_val_acc"] = best_val_acc

    wandb_run.finish()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    app()
