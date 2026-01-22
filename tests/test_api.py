import importlib
import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image


# ============================================================
# Dummy gradio module (doesn't build real UI)
# ============================================================
class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyButton:
    def click(self, fn, inputs=None, outputs=None):
        return None


def _make_dummy_gradio_module() -> ModuleType:
    gr = ModuleType("gradio")

    # Context managers
    gr.Blocks = _DummyCtx
    gr.Column = _DummyCtx
    gr.Row = _DummyCtx

    # UI components
    gr.Markdown = lambda *args, **kwargs: None
    gr.File = lambda *args, **kwargs: object()
    gr.Button = lambda *args, **kwargs: _DummyButton()
    gr.HTML = lambda *args, **kwargs: object()

    return gr


# ============================================================
# Fixture: import dtu_mlops.api safely (no real model and no real gradio)
# ============================================================
@pytest.fixture
def api_module(monkeypatch, tmp_path: Path):
    """
    Import dtu_mlops.api in a safe environment:
    - Create a fake models/resnet18_best.pth so load_model() doesn't crash
    - Patch torch.load to return a dummy checkpoint
    - Patch resnet18 to return a lightweight dummy model
    - Patch gradio so UI build doesn't do anything
    """

    # Work in temp directory so api.py reads models/... from here
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    fake_ckpt = tmp_path / "models" / "resnet18_best.pth"
    fake_ckpt.write_text("dummy", encoding="utf-8")

    # Insert dummy gradio module before importing api.py
    sys.modules["gradio"] = _make_dummy_gradio_module()

    # Patch torch.load used inside load_model()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})

    # Patch the resnet18 factory imported by api.py
    class DummyModel:
        def __init__(self):
            self.loaded = None
            self.eval_called = False
            self.to_device = None

        def load_state_dict(self, sd):
            self.loaded = sd

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.to_device = device
            return self

        def __call__(self, x):
            # return logits with shape (batch, 11)
            return torch.zeros((x.shape[0], 11))

    # Import dtu_mlops.model and patch resnet18 there BEFORE api import
    import dtu_mlops.model as model_module
    monkeypatch.setattr(model_module, "resnet18", lambda **kwargs: DummyModel())

    # Now import api module safely
    if "dtu_mlops.api" in sys.modules:
        del sys.modules["dtu_mlops.api"]

    api = importlib.import_module("dtu_mlops.api")
    return api


# ============================================================
# Tests: load_model()
# ============================================================
def test_load_model_raises_if_missing_file(api_module, tmp_path: Path):
    missing_path = tmp_path / "nope.pth"
    with pytest.raises(FileNotFoundError):
        api_module.load_model(missing_path)


def test_load_model_loads_checkpoint_dict_with_model_state_dict(monkeypatch, api_module, tmp_path: Path):
    ckpt = tmp_path / "x.pth"
    ckpt.write_text("dummy", encoding="utf-8")

    dummy_model = MagicMock()
    dummy_model.eval.return_value = dummy_model
    dummy_model.to.return_value = dummy_model

    monkeypatch.setattr(api_module, "resnet18", lambda **kwargs: dummy_model)

    # checkpoint contains model_state_dict
    monkeypatch.setattr(
        api_module.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {"w": torch.tensor([1.0])}},
    )

    out = api_module.load_model(ckpt)
    dummy_model.load_state_dict.assert_called_once()
    dummy_model.eval.assert_called_once()
    dummy_model.to.assert_called_once()
    assert out is dummy_model


def test_load_model_loads_raw_state_dict(monkeypatch, api_module, tmp_path: Path):
    ckpt = tmp_path / "y.pth"
    ckpt.write_text("dummy", encoding="utf-8")

    dummy_model = MagicMock()
    dummy_model.eval.return_value = dummy_model
    dummy_model.to.return_value = dummy_model

    monkeypatch.setattr(api_module, "resnet18", lambda **kwargs: dummy_model)

    # checkpoint is directly a state_dict (not a dict with model_state_dict)
    monkeypatch.setattr(
        api_module.torch,
        "load",
        lambda *args, **kwargs: {"w": torch.tensor([2.0])},
    )

    out = api_module.load_model(ckpt)
    dummy_model.load_state_dict.assert_called_once_with({"w": torch.tensor([2.0])})
    assert out is dummy_model


# ============================================================
# Tests: preprocessing + labels
# ============================================================
def test_get_preprocessing_pipeline_has_normalize(monkeypatch, api_module):
    """
    Ensure Normalize uses mean/std length equal to n_channels.
    """
    monkeypatch.setattr(
        api_module,
        "INFO",
        {"organamnist": {"n_channels": 1}},
    )

    pipeline = api_module.get_preprocessing_pipeline()

    # transforms.Compose stores its steps in .transforms
    normalize_steps = [t for t in pipeline.transforms if t.__class__.__name__ == "Normalize"]
    assert len(normalize_steps) == 1

    norm = normalize_steps[0]
    assert tuple(norm.mean) == (0.5,)
    assert tuple(norm.std) == (0.5,)


def test_get_class_labels_returns_labels(monkeypatch, api_module):
    monkeypatch.setattr(
        api_module,
        "INFO",
        {"organamnist": {"label": {"0": "class0", "1": "class1"}}},
    )

    labels = api_module.get_class_labels("organamnist")
    assert labels["0"] == "class0"
    assert labels["1"] == "class1"


# ============================================================
# Tests: classify_images()
# ============================================================
def test_classify_images_returns_message_if_none(api_module):
    html = api_module.classify_images(None)
    assert "No images uploaded" in html


def test_classify_images_single_file(monkeypatch, api_module, tmp_path: Path):
    """
    Make sure classify_images:
    - accepts a single image path string
    - returns html containing filename + predicted label
    - embeds image via base64
    """

    # Create a real image file
    img_path = tmp_path / "img.jpg"
    img = Image.new("L", (28, 28))  # grayscale
    img.save(img_path)

    # Patch globals used inside classify_images()
    monkeypatch.setattr(api_module, "preprocess", lambda pil_img: torch.zeros((1, 28, 28)))
    monkeypatch.setattr(api_module, "class_labels", {"0": "class0"})

    class DummyClassifier:
        def __call__(self, x):
            # logits shape (batch, 11)
            # Put highest score at index 0 so argmax -> 0
            out = torch.zeros((x.shape[0], 11))
            out[:, 0] = 10.0
            return out

    monkeypatch.setattr(api_module, "model", DummyClassifier())

    html = api_module.classify_images(str(img_path))

    assert "img.jpg" in html
    assert "class0" in html
    assert "data:image/jpeg;base64," in html
