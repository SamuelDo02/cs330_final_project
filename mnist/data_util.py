import torch
from torchvision import datasets, transforms

def load_data(train=True, transform=None):
    if transform is None:
        # Default transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Load MNIST dataset
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    return data_loader

def load_flipped_data(train=True):
    # Transformations for flipped images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomVerticalFlip(p=1.0)
    ])
    return load_data(train=train, transform=transform)