import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from omegaconf import OmegaConf

import dtu_mlops.evaluate as eval_module

# ============================================================
# Helpers
# ============================================================


def _fake_dataset(num_samples=10, num_classes=3):
    """
    Tiny dataset shaped like MedMNIST:
    - images: (N, 1, 28, 28)
    - labels: (N, 1) because MedMNIST labels have shape (batch_size, 1)
    """
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples, 1))
    return TensorDataset(images, labels)


class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# Unit tests
# ============================================================


def test_compute_metrics():
    """Test metric calculation with known inputs."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 0])  # Last one is wrong
    num_classes = 3

    metrics = eval_module.compute_metrics(y_true, y_pred, num_classes)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision_class_0" in metrics

    # Accuracy should be 4/5 = 0.8
    assert metrics["accuracy"] == 0.8

    # Class 0: True=2, Pred=3 (TP=2, FP=1) -> Precision = 2/3
    # Class 2: True=1, Pred=1 (TP=1, FP=0) -> Precision = 1.0 (if present)


def test_get_predictions_shape_and_mode():
    """Test that get_predictions returns correct shapes and sets eval mode."""
    num_samples = 10
    num_classes = 3
    ds = _fake_dataset(num_samples, num_classes)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = DummyModel(num_classes)
    device = torch.device("cpu")

    # Spy on model.eval()
    with patch.object(model, "eval", wraps=model.eval) as mock_eval:
        preds, labels, probs = eval_module.get_predictions(model, loader, device)

        mock_eval.assert_called()

    assert preds.shape == (num_samples,)
    assert labels.shape == (num_samples,)
    assert probs.shape == (num_samples, num_classes)

    # Check probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)


@patch("dtu_mlops.evaluate.torch.load")
def test_load_model_from_checkpoint(mock_load):
    """Test loading model from checkpoint."""
    num_classes = 3
    in_channels = 1
    device = torch.device("cpu")
    path = Path("dummy.pth")

    # Mock state dict
    mock_model = DummyModel(num_classes)
    mock_load.return_value = {"model_state_dict": mock_model.state_dict()}

    # Mock resnet18 creation
    with patch("dtu_mlops.evaluate.resnet18", return_value=mock_model) as mock_resnet:
        model = eval_module.load_model_from_checkpoint(
            path, "resnet18", num_classes, in_channels, device
        )

        mock_resnet.assert_called_with(num_classes=num_classes, in_channels=in_channels)
        assert model == mock_model
        assert not model.training  # Should be in eval mode due to calls inside


# ============================================================
# Integration tests
# ============================================================


@patch("dtu_mlops.evaluate.wandb")
@patch("dtu_mlops.evaluate.MedMNIST_dataset")
@patch("dtu_mlops.evaluate.load_model_from_checkpoint")
def test_evaluate_runs_end_to_end(mock_load_model, mock_dataset, mock_wandb, tmp_path):
    """
    Test the main evaluate function end-to-end with mocks.
    """
    # Setup mocks
    num_samples = 10
    num_classes = 3
    ds = _fake_dataset(num_samples, num_classes)
    mock_dataset.return_value = ds

    mock_model = DummyModel(num_classes)
    mock_load_model.return_value = mock_model

    # Mock WandB run
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    # Config
    cfg = OmegaConf.create(
        {
            "data_path": str(tmp_path / "data"),
            "data_flag": "organamnist",
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "wandb_entity": "test_entity",
            "wandb_project": "test_project",
            "wandb_name": "test_run",
            "model": {"num_classes": num_classes, "in_channels": 1},
        }
    )

    output_dir = tmp_path / "output"

    # Run evaluation
    metrics = eval_module.evaluate(
        cfg=cfg,
        checkpoint_path=Path("dummy.pth"),
        output_dir=output_dir,
        model_type="resnet18",
    )

    # Assertions
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics

    # Verify WandB logging
    mock_wandb.init.assert_called()
    mock_wandb.log.assert_called()
    mock_run.finish.assert_called()

    # Verify Dataset loaded
    mock_dataset.assert_called()

    # Verify Model loaded
    mock_load_model.assert_called()
