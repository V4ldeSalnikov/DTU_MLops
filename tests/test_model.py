import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dtu_mlops.model import (
    BasicBlock,
    Bottleneck,
    resnet18,
    resnet50,
)
import dtu_mlops.train as train_module

class _DummyPbar:
    """Lightweight tqdm replacement for unit tests.

    Needed because train_epoch/validate uses:
      - for batch in pbar
      - len(pbar)
      - pbar.set_postfix(...)
    """

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable

    def __len__(self):
        return len(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None


def _dummy_tqdm(iterable, desc=None):
    return _DummyPbar(iterable)


def _clone_params(model: nn.Module) -> list[torch.Tensor]:
    """Clone model parameters (cpu tensors) to compare after training/validation."""
    return [p.detach().cpu().clone() for p in model.parameters()]


def _any_param_changed(before: list[torch.Tensor], after: list[torch.Tensor]) -> bool:
    """Return True if any parameter tensor differs."""
    return any(not torch.equal(b, a) for b, a in zip(before, after))


@pytest.fixture(autouse=True)
def _disable_tqdm(monkeypatch):
    """Disable tqdm side effects (progress bars) in training utilities."""
    monkeypatch.setattr(train_module, "tqdm", _dummy_tqdm)


def test_resnet18_construction_has_expected_classifier():
    model = resnet18(num_classes=11, in_channels=1)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 11
    assert model.conv1.in_channels == 1
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)


def test_resnet18_forward_output_shape():
    torch.manual_seed(0)
    model = resnet18(num_classes=11, in_channels=1)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 11)


def test_resnet50_forward_output_shape():
    torch.manual_seed(0)
    model = resnet50(num_classes=11, in_channels=1)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 11)


def test_basicblock_downsample_changes_spatial_and_matches_identity_shape():
    torch.manual_seed(0)

    in_channels = 64
    out_channels = 64
    stride = 2

    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )

    block = BasicBlock(in_channels, out_channels, stride=stride, downsample=downsample)
    x = torch.randn(1, 64, 28, 28)
    y = block(x)

    # With stride=2 we halve spatial dims
    assert y.shape == (1, 64, 14, 14)


def test_bottleneck_expansion_and_output_channels():
    torch.manual_seed(0)

    in_channels = 64
    out_channels = 64
    stride = 1

    downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels * Bottleneck.expansion),
    )

    block = Bottleneck(in_channels, out_channels, stride=stride, downsample=downsample)
    x = torch.randn(2, 64, 28, 28)
    y = block(x)

    # Bottleneck expansion=4 => output channels = out_channels * 4
    assert y.shape == (2, 256, 28, 28)


def test_weight_initialization_batchnorm_is_ones_and_zeros():
    torch.manual_seed(0)
    model = resnet18(num_classes=11, in_channels=1)

    # The model initializes BN weights to 1 and biases to 0
    assert torch.allclose(model.bn1.weight.detach(), torch.ones_like(model.bn1.weight))
    assert torch.allclose(model.bn1.bias.detach(), torch.zeros_like(model.bn1.bias))

    # Conv weights should not be all zeros after kaiming init
    assert not torch.allclose(model.conv1.weight.detach(), torch.zeros_like(model.conv1.weight))


def test_train_epoch_runs_and_updates_parameters():
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Small model for fast unit test
    model = resnet18(num_classes=3, in_channels=1).to(device)

    # Fake data: labels shaped (batch, 1) to match MedMNIST conventions (then squeeze(-1))
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 3, (8, 1))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    params_before = _clone_params(model)

    loss, acc = train_module.train_epoch(
        model=model,
        train_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
    )

    params_after = _clone_params(model)

    # Loss must be finite and accuracy in a valid percentage range
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 100.0

    # Parameters should change after one training epoch
    assert _any_param_changed(params_before, params_after), "Expected model parameters to update after training."


def test_validate_runs_and_does_not_update_parameters():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = resnet18(num_classes=3, in_channels=1).to(device)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 3, (8, 1))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    params_before = _clone_params(model)

    loss, acc = train_module.validate(
        model=model,
        val_loader=loader,
        criterion=criterion,
        device=device,
    )

    params_after = _clone_params(model)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 100.0

    # validate() should not change weights
    assert not _any_param_changed(params_before, params_after), "Model parameters changed during validation()."


def test_train_epoch_uses_train_mode_and_validate_uses_eval_mode():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = resnet18(num_classes=3, in_channels=1).to(device)

    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 3, (4, 1))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train_epoch should set model.train()
    model.eval()
    assert model.training is False
    train_module.train_epoch(model, loader, criterion, optimizer, device, epoch=1)
    assert model.training is True

    # validate should set model.eval()
    model.train()
    assert model.training is True
    train_module.validate(model, loader, criterion, device)
    assert model.training is False
