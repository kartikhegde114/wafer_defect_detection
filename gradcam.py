import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")                     # save pictures to files, do not open windows
import matplotlib.pyplot as plt

# ---------- Settings ----------
IMG = 96                                  # input size the model was trained on
MODEL_FILE = "model_camera.pth"           # the camera fine-tuned model
FOLDER = "Camera_dataset/test"            # camera photos of the test images
SHOW_CLASSES = ["Scratch", "Edge Local"]  # the classes we want to look at
PER_KIND = 3                              # how many right and how many wrong photos per class

# Class names, in the same order the model learned them
with open("split.json") as f:
    classes = json.load(f)["classes"]

# One version of the photos for the model (normalized), one for display (plain colours)
norm_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
show_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])
data = datasets.ImageFolder(FOLDER, transform=norm_tf)
shown = datasets.ImageFolder(FOLDER, transform=show_tf)
assert data.classes == classes, "Class folders do not match the training classes"

# Build ResNet18 with 9 outputs and load what the model learned
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()                              # prediction mode, no learning

# ---------- Grad-CAM: record the activations and gradients of one layer ----------
# layer3 gives a 6x6 map for a 96 pixel input, finer than the last layer (3x3)
store = {}


def forward_hook(module, inputs, output):
    store["act"] = output                                        # what the layer saw
    output.register_hook(lambda grad: store.__setitem__("grad", grad))  # how much each part mattered


model.layer3.register_forward_hook(forward_hook)


def gradcam(x):
    """Return the heatmap (0 to 1), the predicted class and its probability."""
    model.zero_grad()
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())
    logits[0, pred].backward()                                   # gradient of the predicted class
    act = store["act"][0]                                        # shape: channels x height x width
    grad = store["grad"][0]
    weights = grad.mean(dim=(1, 2))                              # importance of each channel
    cam = F.relu((weights[:, None, None] * act).sum(dim=0))      # weighted sum, keep positive parts
    cam = cam.detach().numpy()
    cam = cam / (cam.max() + 1e-8)                               # scale to 0..1
    return cam, pred, float(probs[pred])


# ---------- Pick photos: some right, some wrong, for each class we look at ----------
items = []
for name in SHOW_CLASSES:
    label = classes.index(name)
    right, wrong = [], []
    for i, (path, lab) in enumerate(data.samples):
        if lab != label:
            continue
        x = data[i][0].unsqueeze(0)
        cam, pred, conf = gradcam(x)
        entry = (i, label, pred, conf, cam)
        (right if pred == label else wrong).append(entry)
    items += right[:PER_KIND] + wrong[:PER_KIND]
    print(f"{name}: {len(right)} right and {len(wrong)} wrong photos in total, showing "
          f"{len(right[:PER_KIND])} right and {len(wrong[:PER_KIND])} wrong")

# ---------- Draw: photo and heatmap side by side, two items per row ----------
rows = math.ceil(len(items) / 2)
fig, axes = plt.subplots(rows, 4, figsize=(13, 3.4 * rows))
axes = np.atleast_2d(axes)
for ax in axes.flatten():
    ax.axis("off")                                               # hide empty axes

for n, (i, label, pred, conf, cam) in enumerate(items):
    r, c = n // 2, (n % 2) * 2
    photo = shown[i][0].permute(1, 2, 0).numpy()                 # channels last for drawing
    axes[r][c].imshow(photo)
    axes[r][c].set_title(f"True: {classes[label]}", fontsize=9)
    axes[r][c + 1].imshow(photo)
    axes[r][c + 1].imshow(cam, cmap="jet", alpha=0.45, extent=(0, IMG, IMG, 0),
                          interpolation="bilinear")              # heatmap on top of the photo
    verdict = "right" if pred == label else "WRONG"
    axes[r][c + 1].set_title(f"Model: {classes[pred]} {conf * 100:.0f}% ({verdict})", fontsize=9)

plt.tight_layout()
plt.savefig("gradcam_camera.png", dpi=110)
print("Saved gradcam_camera.png")