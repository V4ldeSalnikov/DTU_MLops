from dtu_mlops.model import Model
from dtu_mlops.data import MedMNIST_dataset

def train():
    dataset = MedMNIST_dataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
