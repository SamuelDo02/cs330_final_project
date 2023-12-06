import os
import argparse
import importlib
import concurrent.futures
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

from util.data_util import load_data, DatasetType  # Ensure DatasetType is imported
import util.net_util as net_util
import models

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
DEVICE = "cpu"
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


def evaluate_checkpoint(model_class, 
                        dataset_type,
                        eval_metadata : EvalMetadata,
                        train_loader, 
                        test_loader, 
                        loss_function):
    checkpoint_path = os.path.join(eval_metadata.checkpoint_dir, eval_metadata.checkpoint_file)
    print(f'Evaluating: {checkpoint_path}')

    # Load model weights from checkpoint
    model = models.init_model(dataset_type, model_class, checkpoint_path, device=DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded model for {eval_metadata.checkpoint_idx}")

    with torch.no_grad():
        train_loss, train_accuracy = evaluate(model, DEVICE, train_loader, loss_function)
        val_loss, val_accuracy = evaluate(model, DEVICE, test_loader, loss_function)

    print(f"{eval_metadata.checkpoint_idx} done")
    return EvalResult(eval_metadata, train_loss, train_accuracy, val_loss, val_accuracy)


def main():
    parser = argparse.ArgumentParser(description='Analyze Model Performance')
    parser.add_argument('--model-info', type=str, nargs='+', required=True, help='Space-separated list of model_class:checkpoint_dir')
    parser.add_argument('--dataset', type=str, choices=['FashionMNIST', 'CIFAR10'], required=True, help='Dataset to use')
    args = parser.parse_args()

    dataset_type = DatasetType[args.dataset]
    loss_function = nn.CrossEntropyLoss()

    metrics = {}

    for model_info in args.model_info:
        model_class, checkpoint_dir = model_info.split(':')

        train_loader = load_data(dataset_type, train=True)
        test_loader = load_data(dataset_type, train=False)

        checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda path: (len(path), path))
        metrics[checkpoint_dir] = [None] * len(checkpoints)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i, checkpoint in enumerate(checkpoints):
                eval_metadata = EvalMetadata(checkpoint_dir, checkpoint, i)
                future = executor.submit(evaluate_checkpoint, 
                                         model_class, 
                                         dataset_type, 
                                         eval_metadata, 
                                         train_loader, 
                                         test_loader, 
                                         loss_function)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                eval_result : EvalResult = future.result()
                metrics[checkpoint_dir][eval_result.metadata.checkpoint_idx] = eval_result

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Generate all checkpoint file paths
            checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda path: (len(path), path))
            futures = [executor.submit(evaluate_checkpoint, 
                                        model_class, 
                                        dataset_type, 
                                        checkpoint_dir, 
                                        checkpoint, 
                                        train_loader, 
                                        test_loader, 
                                        loss_function) for checkpoint in checkpoints]

            for future in concurrent.futures.as_completed(futures):
                future.result()

    # Plotting
    num_epochs = min([metrics[config_str] for config_str in metrics], key=len)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Model Performance on {dataset_type.name}", fontsize=16)

    # Plot training and validation loss for all models
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].train_loss for i in range(num_epochs)]
        plt.plot(epochs, losses, '-o', label=f'{checkpoint_dir}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].train_accuracy for i in range(num_epochs)]
        plt.plot(epochs, accuracies, '-o', label=f'{checkpoint_dir}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].val_loss for i in range(num_epochs)]
        plt.plot(epochs, losses, '-o', label=f'{checkpoint_dir}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].val_accuracy for i in range(num_epochs)]
        plt.plot(epochs, accuracies, '-o', label=f'{checkpoint_dir}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(f"models_performance_{dataset_type.name}.png")

if __name__ == "__main__":
    main()