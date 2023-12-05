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
    normalization_means: tuple
    normalization_stds: tuple


class DatasetType(Enum):
    FashionMNIST = DatasetProperties(
        dataset_class=datasets.FashionMNIST,
        input_size=28 * 28,
        num_classes=10,
        normalization_means=(0.5,),
        normalization_stds=(0.5,)
    )
    CIFAR10 = DatasetProperties(
        dataset_class=datasets.CIFAR10,
        input_size=32 * 32 * 3,
        num_classes=10,
        normalization_means=(0.5, 0.5, 0.5),
        normalization_stds=(0.5, 0.5, 0.5)
    )


def load_data(dataset_type=DatasetType.FashionMNIST, train=True, transform=None, num_workers=os.cpu_count()):
    properties = dataset_type.value
    if transform is None:
        # Default transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(properties.normalization_means, properties.normalization_stds)
        ])

    # Load the specified dataset
    dataset = properties.dataset_class(root='./data', train=train, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    return data_loader


def load_flipped_data(dataset_type=DatasetType.FashionMNIST, train=True):
    properties = dataset_type.value
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.VerticalFlip(),  # Always flip images vertically
        transforms.Normalize(properties.normalization_means, properties.normalization_stds)
    ])

    return load_data(dataset_type=dataset_type, train=train, transform=transform)