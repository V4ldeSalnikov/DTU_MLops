# DTU_MLops

## Goals of the project
The main goal of the project is to do multiclass classification for CT images of different organs.
## Frameworks 
For project template we will use `cookiecutter`. For dependency managment we will use `uv`. For experiments tracking and configuration we will use `hydra` and `weights & biases`. For containerization we will use `docker`. For continious integration we will use `Github actions`. For deployment we will use `Hugging face Spaces`. For linting and codestyle we will use `ruff`linter. For testing we will use `pytest`
## Data 
We are using the dataset - [MedMNIST](https://medmnist.com). From this dataset we will use the OrganAMNIST for training and OrganCMNIST to simulate production.
## Models
We will be primarily using [ResNet-18 and Resnet-50](https://docs.pytorch.org/vision/main/models/resnet.html) implemented in Pytorch
