from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("Wafer_dataset")
image_types = {".jpg", ".jpeg", ".png"}

sizes = Counter()
modes = Counter()
first_images = []

for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in image_types)
    first_images.append((folder.name, files[0]))
    for f in files:
        with Image.open(f) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1

print("Image sizes (width, height) and how many images have each:")
for size, n in sizes.most_common(10):
    print("  ", size, ":", n)

print("\nColour modes:", dict(modes))

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, (name, path) in zip(axes.flatten(), first_images):
    ax.imshow(Image.open(path))
    ax.set_title(name)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png")
print("\nSaved sample_images.png in this folder.")