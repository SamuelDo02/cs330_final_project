import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_util import load_flipped_data
from basic_mlp import MLP, LEARNING_RATE, MOMENTUM, DEVICE, IMAGE_SIZE, CHECKPOINT_INTERVAL_DEFAULT

class InitialReductionMLP(nn.Module):
    def __init__(self, original_model):
        super(InitialReductionMLP, self).__init__()
        # Duplicate the first layer of the original model
        self.duplicated_layer = nn.Linear(IMAGE_SIZE, original_model.fc1.out_features)
        self.duplicated_layer.weight.data = original_model.fc1.weight.data.clone()
        self.duplicated_layer.bias.data = original_model.fc1.bias.data.clone()

        # Use the rest of the model as is and freeze its parameters
        self.model = original_model.model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.view(-1, IMAGE_SIZE)
        x = self.duplicated_layer(x)
        return self.model(x)

def train(train_loader, model, epochs, loss_function, optimizer, checkpoint_dir, checkpoint_interval):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
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
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

def main():
    parser = argparse.ArgumentParser(description='Initial Reduction on Flipped MNIST')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL_DEFAULT, help='Interval for saving model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/initial_reduction', help='Directory for checkpoints')
    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    original_model = MLP()
    original_model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    original_model.to(DEVICE)

    reduction_model = InitialReductionMLP(original_model).to(DEVICE)

    train_loader = load_flipped_data(train=True)
    optimizer = optim.SGD(reduction_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    train(train_loader, reduction_model, args.epochs, loss_function, optimizer, args.checkpoint_dir)

if __name__ == "__main__":
    main()