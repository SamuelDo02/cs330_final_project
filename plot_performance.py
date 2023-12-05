import os
import argparse
import importlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_util import load_data, DatasetType  # Ensure DatasetType is imported
import layer_util


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
    parser = argparse.ArgumentParser(description='Analyze Model Performance on MNIST')
    parser.add_argument('--model-file', type=str, required=True, help='Python file containing the model class (without .py extension)')
    parser.add_argument('--model-class', type=str, required=True, help='Name of the model class in the model file')
    parser.add_argument('--dataset', type=str, choices=['FashionMNIST', 'CIFAR10'], required=True, help='Dataset to use')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing the model checkpoints')
    args = parser.parse_args()

    model_module = importlib.import_module(args.model_file)
    ModelClass = getattr(model_module, args.model_class)

    # Convert dataset argument to enum
    dataset_type = DatasetType[args.dataset.upper()]

    # Generate hidden layer sizes based on dataset properties
    hidden_layers_sizes = layer_util.generate_layer_sizes(dataset_type.value)
    print(hidden_layers_sizes)

    # Load training and test data
    train_loader = load_data(train=True)
    test_loader = load_data(train=False)

    loss_function = nn.CrossEntropyLoss()

    # Lists to store training and validation loss and accuracy
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for checkpoint_file in sorted(os.listdir(args.checkpoint_dir)):
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)
        model = ModelClass(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layers_sizes).to(DEVICE)
        print(f'Evaluated: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        # Evaluate on training data
        train_loss, train_accuracy = evaluate(model, DEVICE, train_loader, loss_function)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)

        # Evaluate on test (validation) data
        val_loss, val_accuracy = evaluate(model, DEVICE, test_loader, loss_function)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)
        
    # Extract the directory name
    checkpoint_dir_name = os.path.basename(os.path.normpath(args.checkpoint_dir))

    # Construct the filename for saving the plot
    plot_filename = f"{checkpoint_dir_name}_{args.model_class}_performance.png"

    # Construct the plot title
    plot_title = f"{args.model_class} Performance on {checkpoint_dir_name}"

    # Plotting
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(12, 10))

    # Update the main title of the plot
    plt.suptitle(plot_title, fontsize=16)

    # Plot training loss and accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_losses, '-o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_accuracies, '-o', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot validation loss and accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, validation_losses, '-o', label='Validation Loss')
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, validation_accuracies, '-o', label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Adjust top to accommodate the main title
    # Save the plot
    plt.savefig(plot_filename)

if __name__ == "__main__":
    main()
