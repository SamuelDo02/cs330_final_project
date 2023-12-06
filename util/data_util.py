from dataclasses import dataclass
from enum import Enum

import torch
from torchvision import datasets, transforms

import os

@dataclass
class DatasetProperties:
    dataset_class: torch.utils.data.Dataset
    input_size: int
    num_classes: int
    transform: transforms.Compose


class DatasetType(Enum):
    FashionMNIST = DatasetProperties(
        dataset_class=datasets.FashionMNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    FashionMNISTTransformed = DatasetProperties(
        dataset_class=datasets.FashionMNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=180),  # Random rotations
            transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),  # Random scaling, keeping image size 28x28
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    CIFAR10 = DatasetProperties(
        dataset_class=datasets.CIFAR10,
        input_size=32 * 32 * 3,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )


def load_data(dataset_type: DatasetType, train=True, num_workers=os.cpu_count()):
    properties = dataset_type.value
    dataset = properties.dataset_class(root='./data', train=train, download=True, transform=properties.transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    return data_loader