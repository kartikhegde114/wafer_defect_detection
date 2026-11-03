import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ---------- Settings ----------
DATA = "Wafer_dataset"
IMG = 96
EPOCHS = 30
BATCH = 32
LR = 0.0003

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------- How images are prepared ----------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ---------- Load the dataset and the SAME split as before ----------
train_full = datasets.ImageFolder(DATA, transform=train_tf)
eval_full = datasets.ImageFolder(DATA, transform=eval_tf)

with open("split.json") as f:
    split = json.load(f)
classes = split["classes"]
train_idx, val_idx = split["train"], split["val"]
print("Train:", len(train_idx), "Validation:", len(val_idx))

train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(Subset(eval_full, val_idx), batch_size=BATCH)

# ---------- Build the model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------- Train ----------
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    scheduler.step()
    train_loss = running_loss / len(train_idx)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            correct += (model(images).argmax(dim=1) == targets).sum().item()
    val_acc = correct / len(val_idx)

    print(f"Epoch {epoch:2d}/{EPOCHS}  train loss: {train_loss:.3f}  validation accuracy: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_v2.pth")

print("\nBest validation accuracy:", round(best_val_acc, 3))
print("Saved the best model as model_v2.pth")