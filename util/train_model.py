import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_util, net_util


# Constants
MOMENTUM = 0.9 
NUM_EPOCHS = 50 
CHECKPOINT_INTERVAL_DEFAULT = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# Models
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers_sizes):
        super(MLP, self).__init__()
        layer_sizes = [input_size] + hidden_layers_sizes + [num_classes]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

        self.model = nn.Sequential(*layers)


    def fine_tune_clear(self):
        for i in range(len(self.model)):
            layer = self.model[i]
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.requires_grad = True

    
    def fine_tune_set_up(self, frozen_range=None):
        if frozen_range == None:
            return

        for i in frozen_range:
            layer = self.model[i]
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.requires_grad = False


    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        return F.log_softmax(self.model(x), dim=1)
    

class LinearReductionMLP(nn.Module):
    def __init__(self, original_model, num_reductions):
        super(LinearReductionMLP, self).__init__()

        new_layers = []
        for layer in original_model.model:
            # Add reduction layer for linear layers
            if isinstance(layer, nn.Linear) and (num_reductions == None or num_reductions > 0):
                new_layer = nn.Linear(layer.in_features, layer.in_features)
                new_layer.weight.data.copy_(torch.eye(layer.in_features)) # identity initialization

                new_layers.append(new_layer)
                if num_reductions != None:
                    num_reductions -= 1
            
            # Freeze original layer 
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.requires_grad = False
            new_layers.append(layer)

        self.model = nn.Sequential(*new_layers)

    
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the image
        return F.log_softmax(self.model(x), dim=1)


# Training
def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Train models on different datasets')

    parser.add_argument('--model-type', type=str, required=True, choices=['MLP', 'LinearReductionMLP'])
    parser.add_argument('--dataset', type=str, required=True, choices=[type.name for type in data_util.DatasetType])
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--load-base-model', type=str, default=None)
    parser.add_argument('--num-reductions', type=int, default=None)
    parser.add_argument('--fine-tune-range',  nargs=2, type=int, default=None)

    args = parser.parse_args()

    # Init dataset and model
    dataset_type = data_util.DatasetType[args.dataset]
    hidden_layer_sizes = net_util.generate_layer_sizes(dataset_type.value)

    if args.model_type == 'MLP':
        model = MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes).to(DEVICE)
    elif args.model_type == 'LinearReductionMLP':
        # Load base model
        base_model = MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes).to(DEVICE)
        if args.load_base_model != None:
            base_model.load_state_dict(torch.load(args.load_base_model, map_location=DEVICE))

        # Init reduction model
        model = LinearReductionMLP(base_model, args.num_reductions).to(DEVICE)

    # Notify of model architecture
    print(model)

    # Load pre-trained model if applicable
    if args.load_model != None:
        model.load_state_dict(torch.load(args.load_model, map_location=DEVICE))

    # Fine-tuning
    if args.model_type == 'MLP' and args.fine_tune_range != None:
        assert args.load_model != None
        start_inc, end_ex = args.fine_tune_range

        if start_inc == end_ex and start_inc == -1:
            model.fine_tune_set_up()
        else:
            model.fine_tune_set_up(frozen_range=range(start_inc, end_ex))

    # Prepare to train model
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader = data_util.load_data(dataset_type)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=MOMENTUM)
    loss_function = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(args.epochs):
        net_util.train(model, DEVICE, train_loader, loss_function, optimizer, epoch + 1, CHECKPOINT_INTERVAL_DEFAULT, args.checkpoint_dir)


if __name__ == "__main__":
    main()