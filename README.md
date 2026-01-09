# DTU_MLops
This is GitHub repository of Group 44 for [DTU 02476 Machine Learning Operation (MLO) Course](https://kurser.dtu.dk/course/02476).
## Goals of the project
The main goal of the project is to build production MLOps pipeline for development of ML project. As for ML goal, it is to do multi-class classification for CT images datasets (COPYPASTE the names!!!) from [MedMNIST](https://medmnist.com). Class labels describe which part of the body is mainly featured in the image.
## Frameworks 
For project template we will use `cookiecutter`. For dependency managment we will use `uv`. For experiments tracking and configuration we will use `hydra` and `weights & biases`. For containerization we will use `docker`. For continious integration we will use `Github actions`. For deployment we will use `Hugging face Spaces`. For linting and codestyle we will use `ruff`linter. For testing we will use `pytest`
## Data 
We are using the dataset - [MedMNIST](https://medmnist.com). From this dataset we will use the OrganAMNIST for training and OrganCMNIST to simulate production.
## Models
We will be primarily using [ResNet-18 and Resnet-50](https://docs.pytorch.org/vision/main/models/resnet.html) implemented in Pytorch
