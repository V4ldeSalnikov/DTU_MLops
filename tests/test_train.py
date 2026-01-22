import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

import dtu_mlops.train as train_module


# ============================================================
# Helpers
# ============================================================
def _clone_params(model: nn.Module):
    """Clone model parameters to check if training updates them."""
    return [p.detach().cpu().clone() for p in model.parameters()]


def _any_param_changed(before, after):
    """True if at least one parameter tensor changed."""
    return any(not torch.equal(b, a) for b, a in zip(before, after))


def _fake_dataset(num_samples=8, num_classes=3):
    """
    Tiny dataset shaped like MedMNIST:
    - images: (N, 1, 28, 28)
    - labels: (N, 1) because your train loop does labels.squeeze(-1)
    """
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples, 1))
    return TensorDataset(images, labels)


class DummyWandbRun:
    """Minimal W&B run object to satisfy wandb.init() usage in train()."""

    def __init__(self):
        self.summary = {}

    def finish(self):
        return None


# ============================================================
# Unit tests: train_epoch and validate
# ============================================================
def test_train_epoch_updates_parameters():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = train_module.resnet18(num_classes=3, in_channels=1).to(device)
    ds = _fake_dataset(num_samples=8, num_classes=3)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = _clone_params(model)
    loss, acc = train_module.train_epoch(model, loader, criterion, optimizer, device, epoch=1)
    after = _clone_params(model)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 100.0

    assert _any_param_changed(before, after), "Expected parameters to change after train_epoch()."


def test_validate_does_not_update_parameters():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = train_module.resnet18(num_classes=3, in_channels=1).to(device)
    ds = _fake_dataset(num_samples=8, num_classes=3)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    before = _clone_params(model)
    loss, acc = train_module.validate(model, loader, criterion, device)
    after = _clone_params(model)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert torch.isfinite(torch.tensor(loss))
    assert 0.0 <= acc <= 100.0

    assert not _any_param_changed(before, after), "validate() must not update parameters."


# ============================================================
# Integration tests: train() with mocking
# ============================================================
@pytest.fixture
def cfg_base(tmp_path):
    """
    Minimal config that passes validate_required_keys() in train().
    NOTE: your train.py requires 'train_samples' key to exist.
    """
    return OmegaConf.create(
        {
            "data_path": str(tmp_path / "data"),
            "data_flag": "organamnist",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "save_best": True,
            "train_samples": None,
            "model_type": "resnet18",
            # optional W&B keys (train() reads them with .get())
            "wandb_entity": "dummy_entity",
            "wandb_project": "dummy_project",
            "wandb_name": "dummy run",
        }
    )


def test_train_runs_end_to_end_with_mocks(monkeypatch, cfg_base):
    """
    This test covers MOST of train():
    - resolves config
    - builds datasets/loaders (mocked)
    - creates model (mocked)
    - runs loop (train_epoch/validate mocked)
    - saves checkpoints (torch.save mocked)
    - model registry functions called (mocked)
    """

    # --- Mock W&B ---
    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: DummyWandbRun())
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)

    # --- Mock datasets ---
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=6, num_classes=3))

    # --- Mock model creation ---
    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    # optimizer needs parameters() to return tensors that require grad
    dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    dummy_model.parameters.return_value = [dummy_param]
    dummy_model.state_dict.return_value = {"w": torch.tensor([1.0])}

    monkeypatch.setattr(train_module, "resnet18", lambda **kwargs: dummy_model)
    monkeypatch.setattr(train_module, "resnet50", lambda **kwargs: dummy_model)

    # --- Mock train_epoch / validate so train loop runs instantly ---
    monkeypatch.setattr(train_module, "train_epoch", lambda *args, **kwargs: (0.50, 60.0))
    monkeypatch.setattr(train_module, "validate", lambda *args, **kwargs: (0.40, 65.0))

    # --- Mock scheduler to cover scheduler.step(val_acc) ---
    class DummyScheduler:
        def __init__(self, optimizer, **kwargs):
            self.steps = []

        def step(self, metric):
            self.steps.append(metric)

    monkeypatch.setattr(train_module.torch.optim.lr_scheduler, "ReduceLROnPlateau", DummyScheduler)

    # --- Mock torch.save ---
    save_mock = MagicMock()
    monkeypatch.setattr(train_module.torch, "save", save_mock)

    # --- Mock model registry functions ---
    monkeypatch.setattr(train_module, "log_artifact", lambda **kwargs: "dummy_artifact")
    monkeypatch.setattr(train_module, "promote_check", lambda **kwargs: False)
    promote_mock = MagicMock()
    monkeypatch.setattr(train_module, "promote_and_demote", promote_mock)

    # --- Run training ---
    train_module.train(cfg=cfg_base)

    # --- Assertions: checkpoint saving happened ---
    saved_paths = [call.args[1] for call in save_mock.call_args_list]
    # run_name = "dummy_run" (spaces replaced by underscore)
    assert any("dummy_run_last.pth" in str(p) for p in saved_paths), "Expected last checkpoint to be saved."
    assert any("dummy_run_best.pth" in str(p) for p in saved_paths), "Expected best checkpoint to be saved."

    # promote_check returns False, so promote_and_demote shouldn't be called
    promote_mock.assert_not_called()


def test_train_subset_branch_applies_when_train_samples_set(monkeypatch, cfg_base):
    """
    Covers the branch where train_samples is not None -> Subset(dataset, indices).
    """
    cfg_base.train_samples = 3

    # W&B mock
    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: DummyWandbRun())
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)

    # dataset size must be > train_samples
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=10, num_classes=3))

    # model mock
    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    dummy_model.parameters.return_value = [dummy_param]
    dummy_model.state_dict.return_value = {"w": torch.tensor([1.0])}
    monkeypatch.setattr(train_module, "resnet18", lambda **kwargs: dummy_model)

    # fast loop
    monkeypatch.setattr(train_module, "train_epoch", lambda *args, **kwargs: (0.5, 60.0))
    monkeypatch.setattr(train_module, "validate", lambda *args, **kwargs: (0.4, 65.0))

    class DummyScheduler:
        def __init__(self, optimizer, **kwargs): pass
        def step(self, metric): pass

    monkeypatch.setattr(train_module.torch.optim.lr_scheduler, "ReduceLROnPlateau", DummyScheduler)

    monkeypatch.setattr(train_module.torch, "save", MagicMock())
    monkeypatch.setattr(train_module, "log_artifact", lambda **kwargs: "dummy_artifact")
    monkeypatch.setattr(train_module, "promote_check", lambda **kwargs: False)
    monkeypatch.setattr(train_module, "promote_and_demote", MagicMock())

    train_module.train(cfg=cfg_base)


def test_train_samples_must_be_positive(monkeypatch, cfg_base):
    """
    Covers:
      if train_samples <= 0: raise ValueError(...)
    """
    cfg_base.train_samples = 0

    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: DummyWandbRun())
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=5, num_classes=3))

    with pytest.raises(ValueError):
        train_module.train(cfg=cfg_base)


def test_train_samples_cannot_exceed_dataset_size(monkeypatch, cfg_base):
    """
    Covers:
      if train_samples > full_size: raise ValueError(...)
    """
    cfg_base.train_samples = 999

    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: DummyWandbRun())
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=5, num_classes=3))

    with pytest.raises(ValueError):
        train_module.train(cfg=cfg_base)


def test_train_raises_on_unknown_model_type(monkeypatch, cfg_base):
    """
    Covers the ValueError branch when model_type is unknown.
    """
    cfg_base.model_type = "not_a_real_model"

    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: DummyWandbRun())
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=5, num_classes=3))

    with pytest.raises(ValueError):
        train_module.train(cfg=cfg_base)


def test_train_promotes_artifact_if_check_true(monkeypatch, cfg_base):
    """
    Covers the model registry promotion branch:
      if promote_check(...): promote_and_demote(...)
    """
    # W&B mock
    wandb_run = DummyWandbRun()
    monkeypatch.setattr(train_module.wandb, "init", lambda **kwargs: wandb_run)
    monkeypatch.setattr(train_module.wandb, "log", lambda *args, **kwargs: None)

    # dataset mock
    monkeypatch.setattr(train_module, "MedMNIST_dataset", lambda **kwargs: _fake_dataset(num_samples=6, num_classes=3))

    # model mock
    dummy_model = MagicMock()
    dummy_model.to.return_value = dummy_model
    dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    dummy_model.parameters.return_value = [dummy_param]
    dummy_model.state_dict.return_value = {"w": torch.tensor([1.0])}
    monkeypatch.setattr(train_module, "resnet18", lambda **kwargs: dummy_model)

    # fast loop
    monkeypatch.setattr(train_module, "train_epoch", lambda *args, **kwargs: (0.5, 60.0))
    monkeypatch.setattr(train_module, "validate", lambda *args, **kwargs: (0.4, 65.0))

    class DummyScheduler:
        def __init__(self, optimizer, **kwargs): pass
        def step(self, metric): pass

    monkeypatch.setattr(train_module.torch.optim.lr_scheduler, "ReduceLROnPlateau", DummyScheduler)
    monkeypatch.setattr(train_module.torch, "save", MagicMock())

    # registry functions: force promote_check True
    monkeypatch.setattr(train_module, "log_artifact", lambda **kwargs: "dummy_artifact")
    monkeypatch.setattr(train_module, "promote_check", lambda **kwargs: True)

    promote_mock = MagicMock()
    monkeypatch.setattr(train_module, "promote_and_demote", promote_mock)

    train_module.train(cfg=cfg_base)

    promote_mock.assert_called_once()
