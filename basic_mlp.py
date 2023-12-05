import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_util import load_data, DatasetType  # Ensure DatasetType is imported
import layer_util

# Constants
LEARNING_RATE = 0.01  # Learning rate
MOMENTUM = 0.9        # Momentum
WEIGHT_DECAY = 0.01   # Weight decay
NUM_EPOCHS = 2000    # Number of epochs
CHECKPOINT_INTERVAL_DEFAULT = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
CHECKPOINT_DIR = "checkpoints"  # Directory for checkpoints

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers, dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(hidden_layers[-1], num_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        return F.log_softmax(self.model(x), dim=1)


# Training function
def train(model, device, train_loader, loss_function, optimizer, epoch, checkpoint_interval, checkpoint_dir):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if batch_idx % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
            torch.save(model.state_dict(), checkpoint_path)

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an MLP on different datasets')
    parser.add_argument('--dataset', type=str, choices=['FashionMNIST', 'CIFAR10'], default='FashionMNIST', help='Dataset to use')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate to use.')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load the pre-trained model')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL_DEFAULT, help='Interval for saving model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR, help='Directory to save checkpoints to')
    args = parser.parse_args()

    # Convert dataset argument to enum
    dataset_type = DatasetType[args.dataset.upper()]

    # Generate hidden layer sizes based on dataset properties
    hidden_layers_sizes = layer_util.generate_layer_sizes(dataset_type.value)

    # Create a dataset-specific checkpoint directory
    dataset_checkpoint_dir = os.path.join(args.checkpoint_dir, f"basic_mlp_lr_{args.learning_rate}_{args.dataset}")
    os.makedirs(dataset_checkpoint_dir, exist_ok=True)

    # Load data
    train_loader = load_data(dataset_type)

    # Initialize the model, optimizer, and loss function
    model = MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layers_sizes).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss()

    # Load pre-trained model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))

    # Run the training
    for epoch in range(NUM_EPOCHS):
        train(model, DEVICE, train_loader, loss_function, optimizer, epoch, args.checkpoint_interval, dataset_checkpoint_dir)


if __name__ == "__main__":
    main()