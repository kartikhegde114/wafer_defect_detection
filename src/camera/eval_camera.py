import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = 96
MODEL_FILE = "model_v2.pth"
FOLDER = "Camera_dataset/test"

with open("split.json") as f:
    classes = json.load(f)["classes"]

tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
data = datasets.ImageFolder(FOLDER, transform=tf)
assert data.classes == classes, "Class folders do not match the training classes"
loader = DataLoader(data, batch_size=32)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

predictions, truths = [], []
with torch.no_grad():
    for images, targets in loader:
        predictions += model(images).argmax(dim=1).tolist()
        truths += targets.tolist()

print("Model:", MODEL_FILE)
print("Camera test images:", len(truths))
print("Accuracy:", round(accuracy_score(truths, predictions) * 100, 1), "%")
print("Macro F1:", round(f1_score(truths, predictions, average="macro"), 3))
print()
print(classification_report(truths, predictions, target_names=classes, digits=3))

cm = confusion_matrix(truths, predictions)
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
plt.savefig("confusion_matrix_camera_before.png")
print("Saved confusion_matrix_camera_before.png")