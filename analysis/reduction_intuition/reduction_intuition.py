# RUN THIS FROM REPO ROOT

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import sys
sys.path.append('util')

import train_model, data_util, net_util

# Load model
dataset_type = data_util.DatasetType.RotatedMNIST

checkpoint_path = 'analysis/reduction_intuition/checkpoints/RotatedMNIST_MLP_REDUCTION/checkpoint_epoch_10_batch_0.pt'

hidden_layer_sizes = net_util.generate_layer_sizes(dataset_type.value)
base_model = train_model.MLP(dataset_type.value.input_size, dataset_type.value.num_classes, hidden_layer_sizes)
model = train_model.LinearReductionMLP(base_model, 1)
model.load_state_dict(torch.load(checkpoint_path))

# Data and inference
train_loader = data_util.load_data(dataset_type)
images, labels = next(iter(train_loader))
image_tensor = images[0]
output_tensor = model.model[0](image_tensor.view(image_tensor.size(0), -1))

# Shape
image_tensor = image_tensor.squeeze(0)
output_tensor = output_tensor.reshape(28, 28)

# Unnormalize
image_tensor = (image_tensor * 0.5) + 0.5
output_tensor = (output_tensor * 0.5) + 0.5
original_tensor = TF.rotate(image_tensor.unsqueeze(0), -90).squeeze(0)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Save side-by-side comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.suptitle('Expected Reduction')

axs[0].imshow(image_tensor.numpy(), cmap='gray')
axs[0].axis('off')

axs[1].imshow(original_tensor.numpy(), cmap='gray')
axs[1].axis('off')

plt.savefig('analysis/reduction_intuition/expected_reduction.png', bbox_inches='tight')

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.suptitle('Actual Reduction')

axs[0].imshow(image_tensor.numpy(), cmap='gray')
axs[0].axis('off')

axs[1].imshow(output_tensor.detach().numpy(), cmap='gray')
axs[1].axis('off')

plt.savefig('analysis/reduction_intuition/actual_reduction.png', bbox_inches='tight')