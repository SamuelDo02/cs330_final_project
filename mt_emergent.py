from dataclasses import dataclass
from typing import List
from enum import Enum, auto
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util.data_util as data_util
import util.net_util as net_util

# Constants
LEARNING_RATE = 0.01
MOMENTUM = 0.9 
NUM_EPOCHS = 100
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
    

# --- EVAL ---
@dataclass
class EvalMetadata:
    checkpoint_dir : str
    checkpoint_file : str
    checkpoint_idx : int


@dataclass
class EvalResult:
    metadata : EvalMetadata

    train_loss : float
    train_accuracy : float
    val_loss : float
    val_accuracy : float


def evaluate(model, mode, device, data_loader, loss_function, subset_size=1000):
    model.eval()
    total_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in data_loader:
            # Evaluate only on a subset
            if count >= subset_size:
                break
            data, target = data.to(device), target.to(device)
            output = model.forward(data, mode)
            total_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
    total_loss /= subset_size
    accuracy = 100. * correct / subset_size
    return total_loss, accuracy


def evaluate_checkpoint(mode,
                        dataset_type,
                        eval_metadata : EvalMetadata,
                        train_loader, 
                        test_loader, 
                        loss_function):
    checkpoint_path = os.path.join(eval_metadata.checkpoint_dir, eval_metadata.checkpoint_file)

    # Load model weights from checkpoint
    hidden_layers_sizes = net_util.generate_layer_sizes(dataset_type.value)
    model = MTEmergentModel(dataset_type.value.input_size, 
                            dataset_type.value.num_classes, 
                            hidden_layers_sizes).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    with torch.no_grad():
        train_loss, train_accuracy = evaluate(model, mode, DEVICE, train_loader, loss_function)
        val_loss, val_accuracy = evaluate(model, mode, DEVICE, test_loader, loss_function)

    return EvalResult(eval_metadata, train_loss, train_accuracy, val_loss, val_accuracy)


def plot_eval(mode_infos,
              dataset_type):
    loss_function = nn.CrossEntropyLoss()

    metrics = {}

    # Ensure no mode errors by running first
    for mode_info in mode_infos:
        mode, checkpoint_dir = mode_info.split(':')
        mode = MTEmergentMode[mode.upper()]

    for mode_info in mode_infos:
        mode, checkpoint_dir = mode_info.split(':')
        mode = MTEmergentMode[mode.upper()]

        train_loader = data_util.load_data(dataset_type, train=True)
        test_loader = data_util.load_data(dataset_type, train=False)

        checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda path: (len(path), path))
        metrics[mode] = [None] * len(checkpoints)

        for i in tqdm(range(len(checkpoints))):
            checkpoint = checkpoints[i]
            eval_metadata = EvalMetadata(checkpoint_dir, checkpoint, i)
            eval_result = evaluate_checkpoint(mode, dataset_type, eval_metadata, train_loader, test_loader, loss_function)
            metrics[mode][i] = eval_result

    # Plotting
    num_epochs = min(len(metrics[config_str]) for config_str in metrics)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Model Performance on {dataset_type.name}", fontsize=16)

    # Plot training and validation loss for all models
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].train_loss for i in range(num_epochs)]
        plt.scatter(epochs, losses, '-o', label=f'{config_str}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].train_accuracy for i in range(num_epochs)]
        plt.scatter(epochs, accuracies, '-o', label=f'{config_str}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].val_loss for i in range(num_epochs)]
        plt.scatter(epochs, losses, '-o', label=f'{config_str}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].val_accuracy for i in range(num_epochs)]
        plt.scatter(epochs, accuracies, '-o', label=f'{config_str}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(f"models_performance_{dataset_type.name}.png")


# --- TRAIN ---
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
    parser.add_argument('--eval', type=str, nargs='+', help='Space-separated list of model:checkpoint_dir')

    args = parser.parse_args()

    # Convert arguments to enums
    dataset_type = data_util.DatasetType[args.dataset]
    mode = MTEmergentMode[args.mode.upper()]

    # Swap to plotting
    if args.eval != None:
        plot_eval(args.eval, dataset_type)
        return

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