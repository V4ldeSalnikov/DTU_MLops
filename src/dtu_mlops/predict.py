"""Inference script for OrganAMNIST ResNet model with Hydra configuration."""

import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig

from dtu_mlops.model import resnet18, resnet50


def load_model_from_hf(
    repo_id: str,
    filename: str,
    model_type: str,
    num_classes: int,
    in_channels: int,
    device: str,
) -> torch.nn.Module:
    """Load trained model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model checkpoint filename
        model_type: Type of model ('resnet18' or 'resnet50')
        num_classes: Number of output classes
        in_channels: Number of input channels
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Downloading model from Hugging Face: {repo_id}/{filename}")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Create model
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    else:
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def load_model_local(
    checkpoint_path: str,
    model_type: str,
    num_classes: int,
    in_channels: int,
    device: str,
) -> torch.nn.Module:
    """Load trained model from local checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('resnet18' or 'resnet50')
        num_classes: Number of output classes
        in_channels: Number of input channels
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from local checkpoint: {checkpoint_path}")
    
    # Create model
    if model_type == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    else:
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(
    image_path: Path,
    target_size: int,
    normalize_mean: float,
    normalize_std: float,
    resize_method: str = "bilinear",
) -> torch.Tensor:
    """Preprocess image for inference with configurable size and normalization.
    
    Args:
        image_path: Path to image file
        target_size: Target image size (will resize to target_size x target_size)
        normalize_mean: Mean for normalization
        normalize_std: Std for normalization
        resize_method: Resizing method ('bilinear', 'nearest', 'bicubic', 'lanczos')
        
    Returns:
        Preprocessed image tensor of shape (1, 1, target_size, target_size)
    """
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Map resize method string to PIL constant
    resize_methods = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resampling = resize_methods.get(resize_method.lower(), Image.Resampling.BILINEAR)
    
    # Resize to target size
    original_size = img.size
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), resampling)
        print(f"  Resized from {original_size} to {img.size} using {resize_method}")
    
    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize using configured statistics
    img_array = (img_array - normalize_mean) / normalize_std
    
    # Convert to tensor and add batch + channel dimensions
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


@hydra.main(version_base=None, config_path="../../configs", config_name="inference")
def predict(cfg: DictConfig) -> None:
    """Run inference on images using Hydra configuration.
    
    The configuration specifies:
    - Model architecture and checkpoint source (HF or local)
    - Image preprocessing settings (size, normalization)
    - Class names and device
    
    Usage:
        # Use default config
        python -m dtu_mlops.predict +images=[image1.png,image2.png]
        
        # Override checkpoint source
        python -m dtu_mlops.predict checkpoint.source=local +images=[img.png]
        
        # Use different model
        python -m dtu_mlops.predict model=resnet50 +images=[img.png]
        
        # Change image size for high-res images
        python -m dtu_mlops.predict image.target_size=224 +images=[img.png]
    """
    # Get image paths from config (passed via +images=[...])
    image_paths = cfg.get("images", [])
    
    if not image_paths:
        print("No image paths provided!")
        print("\nUsage: python -m dtu_mlops.predict +images=[image1.png,image2.png,...]")
        print("   Or: python -m dtu_mlops.predict checkpoint.source=local +images=[img.png]")
        return
    
    # Auto-detect device if not specified
    device = cfg.device
    if device is None or device == "null":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Model: {cfg.model.model_type} ({cfg.model.num_classes} classes, {cfg.model.in_channels} channels)")
    print(f"Image preprocessing: {cfg.image.target_size}x{cfg.image.target_size}, mean={cfg.image.normalize_mean}, std={cfg.image.normalize_std}\n")
    
    # Load model based on source
    if cfg.checkpoint.source == "huggingface":
        model = load_model_from_hf(
            repo_id=cfg.checkpoint.hf_repo,
            filename=cfg.checkpoint.hf_filename,
            model_type=cfg.model.model_type,
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.in_channels,
            device=device,
        )
    else:  # local
        model = load_model_local(
            checkpoint_path=cfg.checkpoint.local_path,
            model_type=cfg.model.model_type,
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.in_channels,
            device=device,
        )
    
    print(f"Model loaded successfully\n")
    
    # Run inference on each image
    with torch.no_grad():
        for img_path_str in image_paths:
            img_path = Path(img_path_str)
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                continue
            
            print(f"Image: {img_path.name}")
            
            # Preprocess image
            img_tensor = preprocess_image(
                image_path=img_path,
                target_size=cfg.image.target_size,
                normalize_mean=cfg.image.normalize_mean,
                normalize_std=cfg.image.normalize_std,
                resize_method=cfg.image.resize_method,
            ).to(device)
            
            # Run inference
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # Print results
            print(f"  Predicted class: {predicted_class} ({cfg.class_names[predicted_class]})")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Top 3 predictions:")
            top3_probs, top3_indices = probabilities[0].topk(3)
            for prob, idx in zip(top3_probs, top3_indices):
                print(f"    - {cfg.class_names[idx.item()]}: {prob:.2%}")
            print()


if __name__ == "__main__":
    predict()
