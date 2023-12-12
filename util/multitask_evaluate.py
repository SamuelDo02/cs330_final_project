import os
import argparse
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
import data_util, net_util
import train_model as models

from tqdm import tqdm

import sys
sys.path.append('mte')
import mte

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_LORA_RANK = 4

@dataclass
class EvalMetadata:
    checkpoint_dir: str
    checkpoint_file: str
    checkpoint_idx: int

@dataclass
class EvalResult:
    metadata: EvalMetadata
    task_results: dict  # Dictionary to store results for each task

def evaluate_multitask(model, device, data_loaders, loss_function, task_indices, max_batches=500):
    model.eval()
    task_results = {task_idx: {'total_loss': 0, 'correct': 0, 'count': 0} for task_idx in task_indices}
    
    with torch.no_grad():
        for task_idx, data_loader in zip(task_indices, data_loaders):
            batch_count = 0
            for data, target in data_loader:
                if max_batches is not None and batch_count >= max_batches:
                    break

                data, target = data.to(device), target.to(device)
                output = model.forward(data, task_idx)
                loss = loss_function(output, target)
                task_results[task_idx]['total_loss'] += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                task_results[task_idx]['correct'] += pred.eq(target.view_as(pred)).sum().item()
                task_results[task_idx]['count'] += len(data)

                batch_count += 1

    for task_idx in task_indices:
        if task_results[task_idx]['count'] > 0:
            task_results[task_idx]['total_loss'] /= task_results[task_idx]['count']
            task_results[task_idx]['accuracy'] = 100. * task_results[task_idx]['correct'] / task_results[task_idx]['count']
    
    return task_results

def evaluate_checkpoint(dataset_type, eval_metadata, train_loaders, test_loaders, loss_function, task_indices):
    checkpoint_path = os.path.join(eval_metadata.checkpoint_dir, eval_metadata.checkpoint_file)

    hidden_layer_sizes = net_util.generate_layer_sizes(dataset_type.value)
    core_model = models.MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes).to(DEVICE)
    model = mte.LoRAMTE(core_model, DEFAULT_LORA_RANK)

    # Initialize LoRA reductions for each task
    for task_id in task_indices:
        model.get_reduction(str(task_id))

    # Load state dict from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Update the model's state dict with the checkpoint
    model_state_dict = model.state_dict()
    model_state_dict.update(checkpoint)  # This assumes the checkpoint contains compatible keys
    model.load_state_dict(model_state_dict)

    train_results = evaluate_multitask(model, DEVICE, train_loaders, loss_function, task_indices)
    val_results = evaluate_multitask(model, DEVICE, test_loaders, loss_function, task_indices)

    return EvalResult(eval_metadata, {'train': train_results, 'val': val_results})

def main():
    parser = argparse.ArgumentParser(description='Evaluate Multitask Model Performance')
    parser.add_argument('--plot-title', type=str, nargs='+', required=True)
    parser.add_argument('--dataset-names', type=str, nargs='+', required=True)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    args = parser.parse_args()

    args.plot_title = ' '.join(args.plot_title)
    loss_function = torch.nn.CrossEntropyLoss()

    # Map dataset names to dataset types and create task indices
    task_datasets = [data_util.DatasetType[dataset_name] for dataset_name in args.dataset_names]
    task_indices = list(range(len(task_datasets)))  # Task indices are based on the order of datasets

    # Load data for each task
    train_loaders = [data_util.load_data(task_dataset, train=True) for task_dataset in task_datasets]
    test_loaders = [data_util.load_data(task_dataset, train=False) for task_dataset in task_datasets]

    checkpoints = sorted(os.listdir(args.checkpoint_dir), key=lambda path: (len(path), path))
    metrics = {}

    for i in tqdm(range(len(checkpoints))):
        checkpoint = checkpoints[i]
        eval_metadata = EvalMetadata(args.checkpoint_dir, checkpoint, i)
        eval_result = evaluate_checkpoint(task_datasets[0], eval_metadata, train_loaders, test_loaders, loss_function, task_indices)
        metrics[checkpoint] = eval_result

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.suptitle(args.plot_title, fontsize=16)

    def plot_metric(split, metric, subplot_index, title):
        plt.subplot(2, 2, subplot_index)
        plt.title(title)

        for task_idx, task_dataset in enumerate(task_datasets):
            metric_data = [metrics[checkpoint].task_results[split][task_idx][metric] for checkpoint in metrics]
            plt.plot(metric_data, '-o', label=f'{task_dataset.name}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss' if 'loss' in metric else 'Accuracy (%)')
        plt.legend()

    # Plot Training Loss
    plot_metric('train', 'total_loss', 1, 'Training Loss Across Tasks')

    # Plot Training Accuracy
    plot_metric('train', 'accuracy', 2, 'Training Accuracy Across Tasks')

    # Plot Validation Loss
    plot_metric('val', 'total_loss', 3, 'Validation Loss Across Tasks')

    # Plot Validation Accuracy
    plot_metric('val', 'accuracy', 4, 'Validation Accuracy Across Tasks')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(f"multitask_model_performance_{args.plot_title}.png")

if __name__ == "__main__":
    main()