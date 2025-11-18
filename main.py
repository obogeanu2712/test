from torchvision.datasets import Caltech256
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

train_dataset = Caltech256(
    root = './',
    transform = ToTensor(),
    target_transform=ToTensor(),
    download=True
)

test_dataset = Caltech256(
    root='./',
    transform=ToTensor(),
    target_transform=ToTensor(),
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

