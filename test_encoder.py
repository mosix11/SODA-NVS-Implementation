
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import GradScaler, autocast


import matplotlib.pyplot as plt
import numpy as np

from src.datasets import MNIST, FashionMNIST, NMR
from src.models import SodaEncoder

from src.utils.nn_utils import get_cpu_device, get_gpu_device

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

gpu = get_gpu_device()
cpu = get_cpu_device()
if gpu == None:
    print("No GPU detected! Using CPU for training!")
    gpu = cpu


# Define hyperparameters
img_size = (64, 64)
num_classes = 3
batch_size = 64
epochs = 20
learning_rate = 0.0005


nmr = NMR(img_size=img_size, num_views=2, batch_size=batch_size, num_workers=12,
          exclude_classes=['02828884', '02933112', '03001627', '03211117', '03636649', '03691459', '04090263', '04379243', '04401088', '04530566'])
model = SodaEncoder(arch='resnet18', img_shape=(3, *img_size), z_dim=num_classes, c_dim=180, c_pos_emb_freq=15)
# model = SodaEncoder(arch='resnet18', img_shape=(3, *img_size), z_dim=num_classes, c_dim=None)

train_loader = nmr.get_train_dataloader()
val_loader = nmr.get_val_dataloader()
test_loader = nmr.get_val_dataloader()

model = model.to(gpu)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

grad_scaler = GradScaler('cuda')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for source_batch, target_batch in train_loader:
        
        views_s, conds_s, labels_s = source_batch
        views_t, conds_t, labels_t = target_batch

        inputs, targets = views_s.to(gpu), labels_s.to(gpu)
        ray_grids = conds_s.to(gpu)
        
        optimizer.zero_grad()
        
        # Enable autocast for forward pass
        with autocast('cuda'):
            # outputs = model(inputs)
            outputs = model(inputs, ray_grids)
            loss = criterion(outputs, targets)
        
        # Scale loss and backpropagate
        grad_scaler.scale(loss).backward()
        # Step optimizer and update scaler
        grad_scaler.step(optimizer)
        grad_scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for source_batch, target_batch in val_loader:
        
            views_s, conds_s, labels_s = source_batch
            views_t, conds_t, labels_t = target_batch
            
            inputs, targets = views_s.to(gpu), labels_s.to(gpu)
            ray_grids = conds_s.to(gpu)
            
            # outputs = model(inputs)
            outputs = model(inputs, ray_grids)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")



# Test Acc
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for source_batch, target_batch in val_loader:
        
        views_s, conds_s, labels_s = source_batch
        views_t, conds_t, labels_t = target_batch
        
        inputs, targets = views_s.to(gpu), labels_s.to(gpu)
        ray_grids = conds_s.to(gpu)
        
        # outputs = model(inputs)
        outputs = model(inputs, ray_grids)
        loss = criterion(outputs, targets)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

val_accuracy = 100. * correct / total
print(f"Test Loss: {val_loss / len(val_loader):.4f}, Test Accuracy: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "resnet18_classification.pth")