from torch.utils.data import Dataset

from dtu_mlops.data import MedMNIST_dataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MedMNIST_dataset("data/raw")
    assert isinstance(dataset, Dataset)
