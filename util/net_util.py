import os
import torch

def generate_layer_sizes(dataset_properties, min_layer_size=64, scale_factor=0.5):
    layer_sizes = []
    layer_size = dataset_properties.input_size

    while True:
        layer_size = int(layer_size * scale_factor)
        if layer_size < min_layer_size or layer_size < dataset_properties.num_classes:
            break
        layer_sizes.append(layer_size)

    return layer_sizes


# Training function
def train(model, 
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
        output = model(data)
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