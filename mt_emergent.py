from typing import List
from enum import Enum, auto
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util.data_util as data_util
import util.net_util as net_util

# Constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9 
NUM_EPOCHS = 500
CHECKPOINT_INTERVAL_DEFAULT = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
CHECKPOINT_DIR = "checkpoints"  # Directory for checkpoints

# --- MTEmergentModel ---
class MTEmergentMode(Enum):
    O = auto(), # Original
    FT = auto(), # Fine tune
    MT = auto(), # Multi task


class MTEmergentModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers_sizes):
        super(MTEmergentModel, self).__init__()
        layer_sizes = [input_size] + hidden_layers_sizes + [num_classes]

        # --- Model and FT layers ---
        self.original_layers = nn.ModuleList([])
        self.ft_layers = nn.ModuleList([])
        
        for i in range(len(layer_sizes) - 1):
            self.original_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.ft_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i]))

            if i != len(layer_sizes) - 2:
                self.original_layers.append(nn.ReLU())
                self.ft_layers.append(None)

        assert len(self.original_layers) == len(self.ft_layers)


    def _disable_grad(self, obj):
        if hasattr(obj, 'requires_grad'):
            obj.requires_grad = False


    def _enable_grad(self, obj):
        if hasattr(obj, 'requires_grad'):
            obj.requires_grad = True


    def _disable_grads(self):
        for layer in self.original_layers:
            self._disable_grad(layer)
        for layer in self.ft_layers:
            self._disable_grad(layer)


    def forward(self, x, mode: MTEmergentMode):
        # Flatten for MLP input
        x = x.view(x.size(0), -1)

        # Prevent cross-path interference
        self._disable_grads()

        if mode == MTEmergentMode.O: # Original
            for layer in self.original_layers:
                self._enable_grad(layer)
                x = layer(x)
        elif mode == MTEmergentMode.FT: # Fine-tune
            for i in range(len(self.original_layers)):
                self._enable_grad(self.ft_layers[i])
                self._enable_grad(self.original_layers[i])

                if self.ft_layers[i] != None:
                    x = self.ft_layers[i](x)
                x = self.original_layers[i](x)
        elif mode == MTEmergentMode.MT: # Multi-task
            pass

        return x


# --- RUN TRAIN ---
def train(model, 
          mode,
          device, 
          train_loader, 
          loss_function, 
          optimizer, 
          epoch, 
          checkpoint_interval, 
          checkpoint_dir, 
          log_percentage=25):
    model.train()
    total_batches = len(train_loader)
    log_interval = int(total_batches * (log_percentage / 100))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data, mode)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if batch_idx % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
            torch.save(model.state_dict(), checkpoint_path)

        # Log
        if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
            processed_data = batch_idx * len(data)
            total_data = len(train_loader.dataset)
            percentage_complete = 100. * batch_idx / total_batches
            current_loss = loss.item()

            print(f"Train Epoch: {epoch} [{processed_data}/{total_data} ({percentage_complete:.0f}%)]\tLoss: {current_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train and Fine-Tune Multi-Task Emergent Model')
    parser.add_argument('--dataset', type=str, choices=[e.name for e in data_util.DatasetType], required=True, help='Dataset to use')
    parser.add_argument('--mode', type=str, choices=['o', 'ft', 'mt'], required=True)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate for model training')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load the pre-trained model')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL_DEFAULT, help='Interval for saving model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR, help='Directory to save checkpoints to')

    args = parser.parse_args()

    # Convert arguments to enums
    dataset_type = data_util.DatasetType[args.dataset]
    mode = MTEmergentMode[args.mode.upper()]

    # Create model
    hidden_layers_sizes = net_util.generate_layer_sizes(dataset_type.value)
    model = MTEmergentModel(dataset_type.value.input_size, 
                            dataset_type.value.num_classes, 
                            hidden_layers_sizes).to(DEVICE)
    
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))

    # Create a dataset-specific checkpoint directory
    dataset_checkpoint_dir = os.path.join(args.checkpoint_dir, f"{mode}_{args.dataset}")
    os.makedirs(dataset_checkpoint_dir, exist_ok=True)

    # Load data
    train_loader = data_util.load_data(dataset_type)

    # Initialize optimizer and argument
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    # Train
    for epoch in range(NUM_EPOCHS):
        train(model, mode, DEVICE, train_loader, loss_function, optimizer, epoch, args.checkpoint_interval, dataset_checkpoint_dir)


if __name__ == '__main__':
    main()