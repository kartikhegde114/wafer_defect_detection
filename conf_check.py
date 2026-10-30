import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------- Settings ----------
IMG = 96                                  # input size the model was trained on
MODEL_FILE = "model_camera.pth"           # the camera fine-tuned model
FOLDER = "Camera_dataset/val"             # validation photos, safe for choosing a limit

# Class names, in the same order the model learned them
with open("split.json") as f:
    classes = json.load(f)["classes"]

# Prepare photos the same way as in training: resize and normalize, no random changes
tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
data = datasets.ImageFolder(FOLDER, transform=tf)
assert data.classes == classes, "Class folders do not match the training classes"
loader = DataLoader(data, batch_size=32)

# Build ResNet18 with 9 outputs and load what the model learned
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

# Collect, for every photo, the model's top confidence and whether it was right
confs, correct = [], []
with torch.no_grad():
    for x, y in loader:
        p = torch.softmax(model(x), dim=1)     # probabilities for the 9 classes
        top_p, top_i = p.max(dim=1)            # best class and its probability
        confs += top_p.tolist()
        correct += (top_i == y).tolist()
confs = np.array(confs)
correct = np.array(correct)

# ---------- Summary ----------
print("Photos:", len(confs), " accuracy:", round(correct.mean() * 100, 1), "%")
print("Average confidence when right:", round(confs[correct].mean(), 2))
print("Average confidence when wrong:", round(confs[~correct].mean(), 2))
print()

# For each possible limit: how many photos are kept, and how accurate are they?
print("Limit -> photos kept -> accuracy on the kept photos")
for limit in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    keep = confs >= limit
    if keep.sum() == 0:
        print(f"  {limit:.1f} -> none kept")
        continue
    print(f"  {limit:.1f} -> {keep.mean() * 100:3.0f}% kept -> {correct[keep].mean() * 100:5.1f}% correct")