import os
import argparse
import importlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

import data_util, net_util
import train_model as models

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


def evaluate_checkpoint(model_type, 
                        dataset_type,
                        eval_metadata : EvalMetadata,
                        train_loader, 
                        test_loader, 
                        loss_function,
                        num_reductions):
    checkpoint_path = os.path.join(eval_metadata.checkpoint_dir, eval_metadata.checkpoint_file)

    # Load model from checkpoint
    hidden_layer_sizes = net_util.generate_layer_sizes(dataset_type.value)

    if model_type == 'MLP':
        model = models.MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes).to(DEVICE)
    elif model_type == 'LinearReductionMLP':
        base_model = models.MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes).to(DEVICE)
        model = models.LinearReductionMLP(base_model, num_reductions).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # Evaluate model
    with torch.no_grad():
        train_loss, train_accuracy = evaluate(model, DEVICE, train_loader, loss_function)
        val_loss, val_accuracy = evaluate(model, DEVICE, test_loader, loss_function)

    return EvalResult(eval_metadata, train_loss, train_accuracy, val_loss, val_accuracy)


def main():
    parser = argparse.ArgumentParser(description='Analyze Model Performance')
    parser.add_argument('--plot-title', type=str, nargs='+', required=True)
    parser.add_argument('--model-info', type=str, nargs='+', required=True, help='Space-separated list of model_type:dataset:label:num_reductions:checkpoint_dir')
    args = parser.parse_args()

    args.plot_title = ' '.join(args.plot_title)
    print(f'Creating plot {args.plot_title}...')

    # Verify that all information is valid
    for model_info in args.model_info:
        model_type, dataset, label, num_reductions, checkpoint_dir = model_info.split(':')
        assert model_type in ['MLP', 'LinearReductionMLP']
        assert dataset in [type.name for type in data_util.DatasetType]
        assert os.path.isdir(checkpoint_dir)
        check = int(num_reductions)

    loss_function = nn.CrossEntropyLoss()
    metrics = {}
    config_to_label = {}

    for model_info in args.model_info:
        model_type, dataset, label, num_reductions, checkpoint_dir = model_info.split(':')

        dataset_type = data_util.DatasetType[dataset]
        train_loader = data_util.load_data(dataset_type, train=True)
        test_loader = data_util.load_data(dataset_type, train=False)

        checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda path: (len(path), path))
        metrics[checkpoint_dir] = [None] * len(checkpoints)

        for i in tqdm(range(len(checkpoints))):
            checkpoint = checkpoints[i]
            eval_metadata = EvalMetadata(checkpoint_dir, checkpoint, i)
            eval_result = evaluate_checkpoint(model_type, dataset_type, eval_metadata, train_loader, test_loader, loss_function, int(num_reductions))
            metrics[checkpoint_dir][i] = eval_result

        config_to_label[checkpoint_dir] = label

    # Plotting
    num_epochs = min(len(metrics[config_str]) for config_str in metrics)
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 10))
    plt.suptitle(args.plot_title, fontsize=16)

    # Plot training and validation loss for all models
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].train_loss for i in range(num_epochs)]
        plt.plot(epochs, losses, '-o', label=config_to_label[config_str])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].train_accuracy for i in range(num_epochs)]
        plt.plot(epochs, accuracies, '-o', label=config_to_label[config_str])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    for config_str in metrics:
        losses = [metrics[config_str][i].val_loss for i in range(num_epochs)]
        plt.plot(epochs, losses, '-o', label=config_to_label[config_str])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    for config_str in metrics:
        accuracies = [metrics[config_str][i].val_accuracy for i in range(num_epochs)]
        plt.plot(epochs, accuracies, '-o', label=config_to_label[config_str])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(f"models_performance_{args.plot_title}.png")

if __name__ == "__main__":
    main()