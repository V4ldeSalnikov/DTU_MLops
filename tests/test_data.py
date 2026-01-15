import pytest
import torch
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from dtu_mlops.data import MedMNIST_dataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MedMNIST_dataset(Path("data/raw"))
    assert isinstance(dataset, Dataset)


@pytest.fixture
def dataset():
    """Fixture to initialize the dataset once for multiple tests."""
    return MedMNIST_dataset(
        data_path=Path("./test_data"),
        data_flag="organamnist",
        split="train",
        data_stat=False,  # Set False for faster testing
    )


def test_dataset_length(dataset):
    """Check that the dataset is not empty."""
    assert len(dataset) > 0, "Dataset should contain at least one sample."


def test_image_and_label_shapes(dataset):
    """
    Requirement: Test different fields (label/image) to
    make sure the image sizes are correct.
    """
    img, label = dataset[0]

    # Check if image is a PyTorch Tensor
    assert isinstance(img, torch.Tensor)

    # MedMNIST images are typically 28x28.
    assert img.shape == (1, 28, 28), f"Expected shape (1, 28, 28), got {img.shape}"

    # Check label shape (usually [1] for MedMNIST classification)
    assert label.shape == (1,), f"Expected label shape (1,), got {label.shape}"


def test_no_empty_labels(dataset):
    """
    Requirement: Ensure there are no empty labels.
    Checking the first 100 samples.
    """
    for i in range(min(100, len(dataset))):
        _, label = dataset[i]

        assert label is not None, f"Label at index {i} is None"

        # Check for NaN using numpy since labels are numpy arrays
        assert not np.isnan(label).any(), f"Label at index {i} contains NaN"


def test_data_normalization(dataset):
    """Verify that images are normalized (roughly between -1 and 1)."""
    img, _ = dataset[0]
    assert (
        img.min() >= -2.0 and img.max() <= 2.0
    ), "Image values are outside expected normalized range."
