import torch
import torch.optim as optim

import os

from itertools import zip_longest

MOMENTUM = 0.9 

def multitask_train(model, 
                    device, 
                    train_loaders, 
                    lr,
                    loss_function, 
                    epoch, 
                    checkpoint_interval, 
                    checkpoint_dir):
    model.train()
    
    batch_idx = 0
    for batches in zip_longest(*train_loaders, fillvalue=None):
        saved = False
        for task_id in range(len(batches)):
            if batches[task_id] == None:
                continue

            data, target = batches[task_id]
            data, target = data.to(device), target.to(device)
            output = model.forward(data, task_id)

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=MOMENTUM)
            optimizer.zero_grad()
            
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            # Save checkpoint
            if not saved and batch_idx % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
                torch.save(model.state_dict(), checkpoint_path)
                saved = True
        
        batch_idx += 1