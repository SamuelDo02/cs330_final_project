import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import argparse
import os

import sys
sys.path.append('util')
import data_util, net_util, multitask_util
import train_model as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
MOMENTUM = 0.9 
DEFAULT_LORA_RANK = 4
CHECKPOINT_INTERVAL_DEFAULT = 100000


class LoRAModule(nn.Module):
    def __init__(self, linear_layer, lora_rank, device):
        super(LoRAModule, self).__init__()
        out_features, in_features = linear_layer.weight.shape
        self.A = nn.Parameter(torch.randn(lora_rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.to(device)

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T


class LoRAMTE(nn.Module):
    def __init__(self, core_model, lora_rank):
        super(LoRAMTE, self).__init__()

        # Freeze all core model layers
        self.core_model = core_model
        for layer in core_model.model:
            if hasattr(layer, 'parameters'):
                for param in layer.parameters():
                    param.requires_grad = False

        # Map to LoRA Reductions
        self.lora_rank = lora_rank
        self.reductions = nn.ModuleDict()

    
    def get_reduction(self, task_id):
        if task_id not in self.reductions: # Not registered yet
            lora_modules = [] # Lora modules for every linear layer
            for layer in self.core_model.model:
                if isinstance(layer, nn.Linear):
                    lora_module = LoRAModule(layer, self.lora_rank, DEVICE)
                    lora_modules.append(lora_module)
                else:
                    lora_modules.append(None) # Placeholder for skipping

            self.reductions[task_id] = nn.ModuleList(lora_modules)

        return self.reductions[task_id]


    def forward(self, x, task_id):
        x = x.view(x.size(0), -1) # Flatten the image

        # Pass through LoRA reductions over original layers
        lora_modules = self.get_reduction(str(task_id))

        for i in range(len(self.core_model.model)):
            layer = self.core_model.model[i]
            output = layer(x)

            # Apply LoRA sum for linear layers
            if lora_modules[i] != None:
                output += lora_modules[i](x)

            x = output

        return F.log_softmax(x, dim=1)


def main():
    # Parse args
    parser = argparse.ArgumentParser(description='Train LoRAMTE on a collection of datasets')

    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--load-core-model', type=str)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    
    args = parser.parse_args()

    # Validate information
    first_dataset_type = data_util.DatasetType[args.datasets[0]]
    input_size = first_dataset_type.value.input_size
    num_classes = first_dataset_type.value.num_classes

    for dataset in args.datasets:
        assert dataset in [type.name for type in data_util.DatasetType]

        dataset_input_size = data_util.DatasetType[args.datasets[0]].value.input_size
        dataset_num_classes = data_util.DatasetType[args.datasets[0]].value.num_classes

        assert input_size == dataset_input_size
        assert num_classes == dataset_num_classes
    
    # Load core model
    hidden_layer_sizes = net_util.generate_layer_sizes(first_dataset_type.value)
    core_model = models.MLP(input_size, num_classes, hidden_layer_sizes).to(DEVICE)

    if args.load_core_model:
        core_model.load_state_dict(torch.load(args.load_core_model, map_location=DEVICE))

    # Construct MTE
    model = LoRAMTE(core_model, DEFAULT_LORA_RANK)

    # Notify of model architecture
    print(model)

    # Prepare to train model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_loaders = []
    for dataset in args.datasets:
        dataset_type = data_util.DatasetType[dataset]
        train_loaders.append(data_util.load_data(dataset_type))

    loss_function = nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.epochs)):
        multitask_util.multitask_train(model, DEVICE, train_loaders, args.lr, loss_function, epoch + 1, CHECKPOINT_INTERVAL_DEFAULT, args.checkpoint_dir)


if __name__ == "__main__":
    main()