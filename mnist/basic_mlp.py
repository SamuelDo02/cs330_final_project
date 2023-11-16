import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_util import load_data

# Constants
IMAGE_SIZE = 28 * 28  # Size of MNIST images
NUM_CLASSES = 10      # Number of classes
HIDDEN_LAYERS_SIZES = [512, 256, 128, 64]  # Sizes of hidden layers
LEARNING_RATE = 0.01  # Learning rate
MOMENTUM = 0.9        # Momentum
NUM_EPOCHS = 10000        # Number of epochs
CHECKPOINT_INTERVAL_DEFAULT = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
CHECKPOINT_DIR = "checkpoints/basic_mlp"  # Directory for checkpoints


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = [nn.Linear(IMAGE_SIZE, HIDDEN_LAYERS_SIZES[0]), nn.ReLU()]
        for i in range(len(HIDDEN_LAYERS_SIZES) - 1):
            layers += [nn.Linear(HIDDEN_LAYERS_SIZES[i], HIDDEN_LAYERS_SIZES[i+1]), nn.ReLU()]
        layers += [nn.Linear(HIDDEN_LAYERS_SIZES[-1], NUM_CLASSES)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)  # Flatten the image
        return F.log_softmax(self.model(x), dim=1)


# Training function
def train(model, device, train_loader, loss_function, optimizer, epoch, checkpoint_interval):
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
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
            torch.save(model.state_dict(), checkpoint_path)

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an MLP on MNIST')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load the pre-trained model')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL_DEFAULT, help='Interval for saving model checkpoints')
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load data
    train_loader = load_data()

    # Initialize the model, optimizer, and loss function
    model = MLP().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    # Load pre-trained model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))

    # Run the training
    for epoch in range(NUM_EPOCHS):
        train(model, DEVICE, train_loader, loss_function, optimizer, epoch, args.checkpoint_interval)


if __name__ == "__main__":
    main()