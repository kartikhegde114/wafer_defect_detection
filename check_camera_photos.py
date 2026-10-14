from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("Camera_dataset/test")
classes = sorted(p.name for p in root.iterdir() if p.is_dir())
PER = 3

fig, axes = plt.subplots(len(classes), PER, figsize=(PER * 2.2, len(classes) * 2.2))
means = {}
all_imgs = []
for r, cls in enumerate(classes):
    files = sorted((root / cls).glob("*.jpg"))
    brightness = []
    for f in files:
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        brightness.append(arr.mean())
        all_imgs.append(arr)
    means[cls] = float(np.mean(brightness))
    for c in range(PER):
        ax = axes[r][c]
        ax.imshow(Image.open(files[c]))
        ax.axis("off")
        if c == 0:
            ax.set_title(cls, fontsize=9, loc="left")
plt.tight_layout()
plt.savefig("camera_samples.png", dpi=100)

print("Average brightness (0 to 255) per class:")
for cls, v in means.items():
    print(f"  {cls:<12}{v:.1f}")

stack = np.stack(all_imgs)
spread = float(np.abs(stack - stack.mean(axis=0)).mean())
print()
print("Average difference between a photo and the overall average photo:", round(spread, 1))
print("Saved camera_samples.png")