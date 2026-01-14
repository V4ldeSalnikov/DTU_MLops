from pathlib import Path

import typer

import medmnist
from medmnist import INFO

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MedMNIST_dataset(Dataset):
    """MedMNIST data wrapper class"""

    def __init__(
            self,
            data_path: Path,
            data_flag: str = "organamnist",
            split: str = "train",
            ) -> None:
        #getting/creating data folder (to avoid downloading dataset each time they are stored locally and just loaded)
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        #retieving dataset info (for example, if data_flag = "organamnist", then organmnist dataset is retireved from MedMNIST dataset collection)
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])
        #getting information for normalization
        output_channels = info["n_channels"] # RGB or Grayscale
        mean = (0.5,) * output_channels
        std = (0.5,) * output_channels
        #transformations of dataset to perform (changing to tensor for PyTorch training and normalization of dataset for better training performance)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        #creating instance of the dataset (this is initialization per say, all code before was just preparation for this)
        self.ds = DataClass(
            root=str(self.data_path),
            split=split,
            download=True,
            transform=transform
            )


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.ds)


    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.ds[index] 

    # def preprocess(self, output_folder: Path) -> None:
    #     """Preprocess the raw data and save it to the output folder."""

# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MedMNIST(data_path)
#     dataset.preprocess(output_folder)


#this fragments basically tests if the dataset class works as intended when .py file is run directly
def main(
        data_path: Path = Path("./MedMNIST_data"),
        data_flag: str = "organamnist",
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True
        ):

    dataset = MedMNIST_dataset(data_path, data_flag = data_flag, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    images, labels = next(iter(loader))
    print(images.shape, labels.shape, len(dataset))

if __name__ == "__main__":
    typer.run(main)
