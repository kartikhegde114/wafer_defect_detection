import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Settings ----------
ROOT = "Camera_dataset"
START = "model_v2.pth"
OUT = "model_camera.pth"
IMG = 96
EPOCHS = 25
BATCH = 32
LR = 0.0001

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("split.json") as f:
    classes = json.load(f)["classes"]

# ---------- How photos are prepared ----------
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=8, translate=(0.12, 0.12),
                            scale=(0.85, 1.15), fill=(110, 105, 110)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    transforms.ToTensor(),
    norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    norm,
])

train_data = datasets.ImageFolder(f"{ROOT}/train", transform=train_tf)
val_data = datasets.ImageFolder(f"{ROOT}/val", transform=eval_tf)
test_data = datasets.ImageFolder(f"{ROOT}/test", transform=eval_tf)
for d in (train_data, val_data, test_data):
    assert d.classes == classes, "Class folders do not match the training classes"

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH)
test_loader = DataLoader(test_data, batch_size=BATCH)
print("Camera photos  train:", len(train_data), " val:", len(val_data), " test:", len(test_data))


def make_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(torch.load(START, map_location="cpu"))
    return m.to(device)


def predict(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(device)).argmax(dim=1).tolist()
            truths += y.tolist()
    return truths, preds


# ---------- Before: the clean-trained model on camera test photos ----------
model = make_model()
t, p = predict(model, test_loader)
print(f"\nBEFORE fine-tuning ({START}) on camera test photos:")
print("  Accuracy:", round(accuracy_score(t, p) * 100, 1), "%   Macro F1:",
      round(f1_score(t, p, average="macro"), 3))

# ---------- Fine-tune ----------
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val = 0.0
print()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    scheduler.step()

    vt, vp = predict(model, val_loader)
    val_acc = accuracy_score(vt, vp)
    print(f"Epoch {epoch:2d}/{EPOCHS}  train loss: {running / len(train_data):.3f}  "
          f"camera validation accuracy: {val_acc:.3f}")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), OUT)

print("\nBest camera validation accuracy:", round(best_val, 3))

# ---------- After: best model on camera test photos, once ----------
model.load_state_dict(torch.load(OUT, map_location=device))
t, p = predict(model, test_loader)
print(f"\nAFTER fine-tuning ({OUT}) on camera test photos:")
print("  Accuracy:", round(accuracy_score(t, p) * 100, 1), "%   Macro F1:",
      round(f1_score(t, p, average="macro"), 3))
print()
print(classification_report(t, p, target_names=classes, digits=3, zero_division=0))

cm = confusion_matrix(t, p)
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("confusion_matrix_camera_after.png")
print("Saved confusion_matrix_camera_after.png and", OUT)