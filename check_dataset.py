from pathlib import Path

root = Path("Wafer_dataset")
image_types = {".jpg", ".jpeg", ".png"}

grand_total = 0
for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    count = sum(1 for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in image_types)
    subfolders = [p.name for p in folder.iterdir() if p.is_dir()]
    grand_total += count
    print(folder.name, ":", count, "images", "| subfolders:", subfolders)

print("\nGRAND TOTAL :", grand_total)