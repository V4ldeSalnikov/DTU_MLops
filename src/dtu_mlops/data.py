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
            data_stat: bool = True,
            ) -> None:
        #getting/creating data folder (to avoid downloading dataset each time they are stored locally and just loaded)
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        #retieving dataset info (for example, if data_flag = "organamnist", then organmnist dataset is retireved from MedMNIST dataset collection)
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])
        #getting information for normalization
        if (data_stat):
            #temporary dataset instance is created to compute mean and std of dataset for normalization (in this case needed dataset is downloaded only once, in next step it will be loaded from local storage)
            transform_temp = transforms.Compose([
                transforms.ToTensor(),
            ])
            ds_temp = DataClass(
                root=str(self.data_path),
                split=split,
                download=True,
                transform=transform_temp
                )
            #mean and std are computed from the dataset
            mean, std = MedMNIST_dataset.compute_mean_std(ds_temp)
        else:
            #using default mean and std values for normalization
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

    def compute_mean_std(dataset):
        #dataset is put into loader to process in batches (whole dataset might not fit into memory)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        #counting variables are created (they are updated in each iteration of the loop and form basis for further calculations)
        mean = 0.0
        std = 0.0
        total_images = 0
        #iterating through the dataset
        for images, _ in loader:
            #getting number of images in the current batch (in last iteration it might be smaller than batch size)
            n_img_batch = images.size(0)
            #flattening images to calculate mean and std per channel(in case of this project images are grayscale, so n_channels = 1, but in MedMNIST dataset there are also RGB images, so for future compatibility this is implemented this way)
            n_channels = images.size(1)
            images = images.view(n_img_batch, n_channels, -1)
            #calculating mean and std for current iteration and adding them to count variables. Mean.std are calculated per channel and then they are summed over all images in the batch
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            #updating total number of images processed so far
            total_images += n_img_batch
        #mean and std of dataset are calculated (as in loop means and stds were sums over all images, now they are need to be divided by number of those images to get actual mean and std)
        mean /= total_images
        std /= total_images
        #mean and std of dataset is returned (before returning they are converted from tensors to lists for easier handling in intended use case (transforms.Normalize())
        return mean.tolist(), std.tolist()


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
        data_stat: bool = True,
        batch_size: int = 64,
        shuffle: bool = True
        ):

    dataset = MedMNIST_dataset(data_path, data_flag = data_flag, split=split,data_stat = data_stat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    images, labels = next(iter(loader))
    print(images.shape, labels.shape, len(dataset))

if __name__ == "__main__":
    typer.run(main)
