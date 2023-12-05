import os
import argparse
import importlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from util.data_util import load_data, DatasetType  # Ensure DatasetType is imported
import util.net_util as net_util
import models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
VALIDATION_SUBSET_SIZE = 1000

def evaluate(model, device, data_loader, loss_function, subset_size=VALIDATION_SUBSET_SIZE):
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
            output = model(data)
            total_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
    total_loss /= subset_size
    accuracy = 100. * correct / subset_size
    return total_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Analyze Model Performance')
    parser.add_argument('--model-info', type=str, nargs='+', required=True, help='Space-separated list of model_class:checkpoint_dir')
    parser.add_argument('--dataset', type=str, choices=['FashionMNIST', 'CIFAR10'], required=True, help='Dataset to use')
    args = parser.parse_args()

    dataset_type = DatasetType[args.dataset]
    loss_function = nn.CrossEntropyLoss()

    # Store metrics for all models
    all_training_losses = {}
    all_validation_losses = {}
    all_training_accuracies = {}
    all_validation_accuracies = {}

    for model_info in args.model_info:
        model_class, checkpoint_dir = model_info.split(':')

        train_loader = load_data(dataset_type, train=True)
        test_loader = load_data(dataset_type, train=False)

        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []

        for checkpoint_file in sorted(os.listdir(checkpoint_dir), key=lambda path: (len(path), path)):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f'Evaluated: {checkpoint_path}')

            # Load model weights from checkpoint
            model = models.init_model(dataset_type, model_class, checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

            with torch.no_grad():
                train_loss, train_accuracy = evaluate(model, DEVICE, train_loader, loss_function)
                training_losses.append(train_loss)
                training_accuracies.append(train_accuracy)

                val_loss, val_accuracy = evaluate(model, DEVICE, test_loader, loss_function)
                validation_losses.append(val_loss)
                validation_accuracies.append(val_accuracy)

        # Store metrics for the current model
        all_training_losses[checkpoint_dir] = training_losses
        all_validation_losses[checkpoint_dir] = validation_losses
        all_training_accuracies[checkpoint_dir] = training_accuracies
        all_validation_accuracies[checkpoint_dir] = validation_accuracies

    # Plotting
    epochs = range(1, len(next(iter(all_training_losses.values()))) + 1)
    plt.figure(figsize=(12, 10))
    plt.suptitle('Model Performance Comparison', fontsize=16)

    # Plot training and validation loss for all models
    plt.subplot(2, 2, 1)
    for checkpoint_dir, losses in all_training_losses.items():
        plt.plot(epochs, losses, '-o', label=f'{checkpoint_dir} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for checkpoint_dir, accuracies in all_training_accuracies.items():
        plt.plot(epochs, accuracies, '-o', label=f'{checkpoint_dir} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    for checkpoint_dir, losses in all_validation_losses.items():
        plt.plot(epochs, losses, '-o', label=f'{checkpoint_dir} Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    for checkpoint_dir, accuracies in all_validation_accuracies.items():
        plt.plot(epochs, accuracies, '-o', label=f'{checkpoint_dir} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig('models_performance_comparison.png')

if __name__ == "__main__":
    main()