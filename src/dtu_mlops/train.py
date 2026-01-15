import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
import typer
from tqdm import tqdm

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
    data_path: Path = Path("data/medmnist"),
    data_flag: str = "organamnist",
    model_type: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: Optional[str] = None,
    checkpoint_dir: Path = Path("models/checkpoints"),
    save_best: bool = True,
) -> None:
    """Train ResNet model on MedMNIST dataset
    
    Args:
        data_path: Path to data directory
        data_flag: MedMNIST dataset flag (e.g., "organamnist", "pathmnist")
        model_type: Model architecture ("resnet18" or "resnet50")
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        num_workers: Number of workers for data loading
        device: Device to train on (None for auto-detect)
        checkpoint_dir: Directory to save model checkpoints
        save_best: Whether to save best model based on validation accuracy
    """
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
            print(f"✓ Saved best model with validation accuracy: {val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    app()
