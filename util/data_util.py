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
    MNIST = DatasetProperties(
        dataset_class=datasets.MNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    RotatedMNIST = DatasetProperties(
        dataset_class=datasets.MNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((90, 90)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    SVHN = DatasetProperties(
        dataset_class=datasets.SVHN,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    KMNIST = DatasetProperties(
        dataset_class=datasets.KMNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )
    FashionMNIST = DatasetProperties(
        dataset_class=datasets.FashionMNIST,
        input_size=28 * 28,
        num_classes=10,
        transform=transforms.Compose([
            transforms.ToTensor(),
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

    if dataset_type in [DatasetType.SVHN]:
        split = 'train' if train else 'test'
        dataset = properties.dataset_class(root='./data', split=split, download=True, transform=properties.transform)
    else:
        dataset = properties.dataset_class(root='./data', train=train, download=True, transform=properties.transform)
        
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    return data_loader