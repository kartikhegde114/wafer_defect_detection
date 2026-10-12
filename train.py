import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

# ---------- Settings ----------
DATA = "Wafer_dataset"
IMG = 64
EPOCHS = 15
BATCH = 32

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

# ---------- Load the dataset ----------
train_full = datasets.ImageFolder(DATA, transform=train_tf)
eval_full = datasets.ImageFolder(DATA, transform=eval_tf)
classes = train_full.classes
labels = train_full.targets
print("Classes:", classes)

# ---------- Split into train / validation / test ----------
all_idx = list(range(len(labels)))
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.30, stratify=labels, random_state=42)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)

print("Train:", len(train_idx), "Validation:", len(val_idx), "Test:", len(test_idx))

with open("split.json", "w") as f:
    json.dump({"classes": classes, "train": train_idx,
               "val": val_idx, "test": test_idx}, f)

train_loader = DataLoader(Subset(train_full, train_idx), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(Subset(eval_full, val_idx), batch_size=BATCH)

# ---------- Build the model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    train_loss = running_loss / len(train_idx)

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == targets).sum().item()
    val_acc = correct / len(val_idx)

    print(f"Epoch {epoch:2d}/{EPOCHS}  train loss: {train_loss:.3f}  validation accuracy: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_best.pth")

print("\nBest validation accuracy:", round(best_val_acc, 3))
print("Saved the best model as model_best.pth")