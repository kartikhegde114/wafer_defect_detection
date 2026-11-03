import json
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------- Settings ----------
IMG = 96                                   # input size the models were trained on
MODEL_FILES = ["model_v2.pth", "model_camera.pth"]   # the two models to test

# Class names, in the same order the models learned them
with open("split.json") as f:
    classes = json.load(f)["classes"]

# Numbers used to scale pixels, the same as in training
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------- Fake pictures that contain NO wafer ----------
rng = np.random.default_rng(0)             # fixed seed so the noise is the same every run
tests = {
    "black": np.zeros((IMG, IMG, 3), dtype=np.float32),
    "dark grey": np.full((IMG, IMG, 3), 0.2, dtype=np.float32),
    "mid grey": np.full((IMG, IMG, 3), 0.5, dtype=np.float32),
    "white": np.ones((IMG, IMG, 3), dtype=np.float32),
    "random noise": rng.random((IMG, IMG, 3)).astype(np.float32),
}

for model_file in MODEL_FILES:
    # Skip a model file that is missing instead of crashing
    if not os.path.exists(model_file):
        print(f"\n{model_file} not found, skipping.")
        continue

    # Build ResNet18 with 9 outputs and load what the model learned
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()                           # prediction mode, no learning

    print(f"\nModel: {model_file}")
    for name, img in tests.items():
        # Scale the picture like in training and turn it into a tensor
        x = (img - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

        # Ask the model for the probability of each class
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[0].numpy()

        # Show the three most likely classes with their percentages
        top = np.argsort(p)[::-1][:3]
        result = ", ".join(f"{classes[i]} {p[i] * 100:.0f}%" for i in top)
        print(f"  {name:13s} -> {result}")