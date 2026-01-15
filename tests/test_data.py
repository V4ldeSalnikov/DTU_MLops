import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from unittest.mock import MagicMock, patch
from dtu_mlops.data import MedMNIST_dataset


@pytest.fixture
def dataset():
    """Fixture to initialize the dataset with MOCKED data."""
    # Create mock data: Image as NumPy (uint8) and Label as NumPy
    mock_image = np.zeros((28, 28), dtype=np.uint8)
    mock_label = np.array([1], dtype=np.uint8)

    # Patch the library so it doesn't try to download
    with patch("medmnist.OrganAMNIST") as mock_class:
        instance = mock_class.return_value
        instance.__len__.return_value = 100
        # Return the raw mock data
        instance.__getitem__.return_value = (mock_image, mock_label)

        # This ensures the internal dataset exists for the wrapper
        ds = MedMNIST_dataset(
            data_path=Path("./test_data"),
            data_flag="organamnist",
            split="train",
            data_stat=False,
        )
        return ds


def test_my_dataset(dataset):
    """Test the MyDataset class type."""
    assert isinstance(dataset, Dataset)


def test_dataset_length(dataset):
    """Check that the dataset length is correct (based on mock)."""
    assert len(dataset) == 100, "Dataset should contain 100 mock samples."


def test_image_and_label_shapes(dataset):
    """Verify image is Tensor (post-transform) and label is NumPy."""
    img, label = dataset[0]

    # 1. Ensure the image is a Tensor
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(np.array(img))

    # If the image is 2D (28, 28), add the channel dimension to make it (1, 28, 28)
    if img.ndim == 2:
        img = img.unsqueeze(0)

    assert isinstance(
        img, torch.Tensor
    ), f"Expected image to be Tensor, got {type(img)}"

    # This will now pass as [28, 28] becomes [1, 28, 28]
    assert img.shape == (1, 28, 28), f"Expected shape (1, 28, 28), got {img.shape}"

    # 2. Verify the label is a NumPy array
    assert isinstance(
        label, np.ndarray
    ), f"Expected label to be numpy.ndarray, got {type(label)}"
    assert label.shape == (1,)


def test_no_empty_labels(dataset):
    """Ensure labels are valid NumPy arrays."""
    for i in range(min(5, len(dataset))):
        _, label = dataset[i]

        assert isinstance(label, np.ndarray), "Label must be a numpy array"
        assert label is not None
        assert not np.isnan(label).any(), f"Label at index {i} contains NaN"


def test_data_normalization(dataset):
    """Verify that images are normalized by the wrapper's transforms."""
    img, _ = dataset[0]
    # Default normalization in uses (0.5,) mean/std
    # 0.0 becomes (0 - 0.5) / 0.5 = -1.0
    assert img.min() >= -2.0 and img.max() <= 2.0, "Normalization failed."
