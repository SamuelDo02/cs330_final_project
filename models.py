import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util.data_util import load_data, DatasetType
import util.net_util as net_util

# Constants
LEARNING_RATE = 0.01  # Learning rate
MOMENTUM = 0.9        # Momentum
WEIGHT_DECAY = 0.01   # Weight decay
NUM_EPOCHS = 500    # Number of epochs
CHECKPOINT_INTERVAL_DEFAULT = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device configuration
CHECKPOINT_DIR = "checkpoints"  # Directory for checkpoints


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers):
        super(MLP, self).__init__()
        self.num_hidden_layers = 0

        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU()]
            self.num_hidden_layers += 1
        layers += [nn.Linear(hidden_layers[-1], num_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        return F.log_softmax(self.model(x), dim=1)
    

class ReductionMLP(nn.Module):
    def __init__(self, original_model):
        super(ReductionMLP, self).__init__()
        self.num_hidden_layers = 0

        new_layers = []
        for layer in original_model.model:
            if isinstance(layer, nn.Linear):
                # Duplicate the layer
                new_layer = nn.Linear(layer.in_features, layer.in_features)

                # Freeze original layer weights
                for param in layer.parameters():
                    param.requires_grad = False

                # Add duplicated layer before the original layer
                new_layers.extend([new_layer, layer])
                self.num_hidden_layers += 2
            else:
                # For non-linear layers, just append
                new_layers.append(layer)

        self.model = nn.Sequential(*new_layers)

    
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        return F.log_softmax(self.model(x), dim=1)
    

def init_model(dataset_type: DatasetType, 
               model_str: str, 
               load_model: str = None, 
               load_base_model: str = None,
               device: str = DEVICE):
    assert model_str in ['MLP', 'ReductionMLP']

    # Init model
    if model_str == 'MLP':
        hidden_layers_sizes = net_util.generate_layer_sizes(dataset_type.value)
        model = MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layers_sizes).to(device)
    elif model_str == 'ReductionMLP':
        hidden_layers_sizes = net_util.generate_layer_sizes(dataset_type.value)
        original_model = MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layers_sizes)
        
        if load_base_model: # Load required base model if no pre-trained model
            original_model.load_state_dict(torch.load(load_base_model, map_location=device))
            original_model.to(device)
        elif not load_model:
            raise Exception('Base MLP not provided.')
        
        model = ReductionMLP(original_model).to(device)

    print('Before loadingx')
        
    # Load pre-trained model if specified
    if load_model:
        model.load_state_dict(torch.load(load_model, map_location=device))

    return model


def add_common_arguments(parser):
    parser.add_argument('--dataset', type=str, choices=['FashionMNIST', 'CIFAR10'], default='FashionMNIST', help='Dataset to use')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate for model training')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load the pre-trained model')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL_DEFAULT, help='Interval for saving model checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR, help='Directory to save checkpoints to')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train models on different datasets')
    subparsers = parser.add_subparsers(dest='model', required=True)

    # Subparser for MLP model
    mlp_parser = subparsers.add_parser('MLP', help='MLP model')
    add_common_arguments(mlp_parser)

    # Subparser for ReductionMLP model
    reduction_mlp_parser = subparsers.add_parser('ReductionMLP', help='ReductionMLP model')
    add_common_arguments(reduction_mlp_parser)
    reduction_mlp_parser.add_argument('--load-base-model', type=str, default=None, help='Path to load pre-trained model as base model')

    args = parser.parse_args()

    # Convert dataset argument to enum
    dataset_type = DatasetType[args.dataset]

    # Init model
    model = init_model(dataset_type, args.model, args.load_model, args.load_base_model)

    # Create a dataset-specific checkpoint directory
    dataset_checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.model}_hl-{model.num_hidden_layers}_lr-{args.learning_rate}_{args.dataset}")
    os.makedirs(dataset_checkpoint_dir, exist_ok=True)

    # Load data
    train_loader = load_data(dataset_type)

    # Initialize optimizer and argument
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    # Train
    for epoch in range(NUM_EPOCHS):
        net_util.train(model, DEVICE, train_loader, loss_function, optimizer, epoch, args.checkpoint_interval, dataset_checkpoint_dir)


if __name__ == "__main__":
    main()