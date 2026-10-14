import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

import ctypes
import json
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from torchvision import datasets

# ---------- Settings ----------
URL = "rtsp://172.32.0.93/live/0"
BOX_FRACTION = 0.6
SAVE_SIZE = 128
SETTLE = 0.7
OUT = Path("Camera_dataset")
WIN = "Capture"

ctypes.windll.user32.SetProcessDPIAware()
SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)


class LatestFrame:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.stamp = 0.0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(2)
                continue
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
                    self.stamp = time.time()
            cap.release()

    def get(self):
        with self.lock:
            return self.frame, self.stamp


def box_coords(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * BOX_FRACTION)
    return (w - side) // 2, (h - side) // 2, side


def make_preview(frame):
    x1, y1, side = box_coords(frame)
    shown = frame.copy()
    cv2.rectangle(shown, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 6)
    scale = 480 / shown.shape[1]
    return cv2.resize(shown, (480, int(shown.shape[0] * scale)))


def make_canvas(wafer, ratio, preview=None):
    canvas = np.full((SCREEN_H, SCREEN_W, 3), 60, dtype=np.uint8)
    size = int(min(SCREEN_H, SCREEN_W) * ratio)
    big = cv2.resize(wafer, (size, size), interpolation=cv2.INTER_NEAREST)
    y = (SCREEN_H - size) // 2
    x = (SCREEN_W - size) // 2
    canvas[y:y + size, x:x + size] = big
    if preview is not None:
        canvas[0:preview.shape[0], 0:preview.shape[1]] = preview
    return canvas


def finish():
    cv2.destroyAllWindows()
    raise SystemExit


# ---------- Build the list of images to photograph ----------
data = datasets.ImageFolder("Wafer_dataset")
with open("split.json") as f:
    split = json.load(f)
classes = split["classes"]

jobs = []
for part in ("train", "val", "test"):
    for idx in split[part]:
        path, label = data.samples[idx]
        jobs.append((part, idx, path, classes[label]))
random.seed(0)
random.shuffle(jobs)
print("Images to photograph:", len(jobs))

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
grabber = LatestFrame(URL)

# ---------- Step A: line up the wafer with the green box ----------
ratio = 0.6
sample = cv2.imread(jobs[0][2])
while True:
    frame, _ = grabber.get()
    preview = make_preview(frame) if frame is not None else None
    cv2.imshow(WIN, make_canvas(sample, ratio, preview))
    key = cv2.waitKey(30) & 0xFF
    if key == ord("w"):
        ratio = min(ratio + 0.03, 0.95)
    elif key == ord("s"):
        ratio = max(ratio - 0.03, 0.2)
    elif key == 32:
        break
    elif key == ord("q"):
        finish()

# ---------- Calibration: find the wafer area in the camera picture ----------
def calibrate():
    size = int(min(SCREEN_H, SCREEN_W) * ratio)
    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    top = (SCREEN_H - size) // 2
    left = (SCREEN_W - size) // 2
    canvas[top:top + size, left:left + size] = 255
    cv2.imshow(WIN, canvas)
    t = time.time()
    while time.time() - t < 1.5:
        cv2.waitKey(10)

    frame, _ = grabber.get()
    if frame is None:
        print("No camera frame.")
        return None

    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No bright square found.")
        return None

    biggest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(biggest)
    fh, fw = frame.shape[:2]

    inside = float(gray[by:by + bh, bx:bx + bw].mean())
    outside = float(gray[mask == 0].mean()) if (mask == 0).any() else inside
    aspect = bw / max(bh, 1)
    area_share = (bw * bh) / (fw * fh)
    print(f"Square found: x={bx} y={by} w={bw} h={bh} in a {fw}x{fh} picture")
    print(f"Contrast inside vs outside: {inside - outside:.0f}, aspect {aspect:.2f}, area share {area_share:.2f}")

    touches_edge = bx <= 2 or by <= 2 or bx + bw >= fw - 2 or by + bh >= fh - 2
    if inside - outside < 40 or not (0.75 <= aspect <= 1.33) or not (0.04 <= area_share <= 0.9) or touches_edge:
        print("The square is not clearly and completely visible.")
        print("Move the camera so the whole white square is in the picture, away from the edges.")
        return None
    return bx, by, bw, bh, fw, fh, frame


result = calibrate()
if result is None:
    finish()
bx, by, bw, bh, fw, fh, calib_frame = result

side = int(min(bw, bh) * 0.96)
cx1 = int(np.clip(bx + bw // 2 - side // 2, 0, fw - side))
cy1 = int(np.clip(by + bh // 2 - side // 2, 0, fh - side))

with open("crop_box.json", "w") as f:
    json.dump({"x": cx1, "y": cy1, "side": side, "frame_w": fw, "frame_h": fh}, f)

check = calib_frame.copy()
cv2.rectangle(check, (cx1, cy1), (cx1 + side, cy1 + side), (0, 255, 0), 3)
cv2.imwrite("calibration_check.jpg", check)
print("Calibration done. Crop box saved in crop_box.json. See calibration_check.jpg")

# ---------- Photograph every image ----------
start = time.time()
for n, (part, idx, path, cls) in enumerate(jobs, 1):
    target = OUT / part / cls / f"{idx}.jpg"
    if target.exists():
        continue

    cv2.imshow(WIN, make_canvas(cv2.imread(path), ratio))
    t_show = time.time()
    while time.time() - t_show < SETTLE:
        if cv2.waitKey(10) & 0xFF == ord("q"):
            finish()

    frame, stamp = grabber.get()
    waited = time.time()
    while (frame is None or stamp < t_show + SETTLE) and time.time() - waited < 5:
        cv2.waitKey(10)
        frame, stamp = grabber.get()
    if frame is None or frame.shape[:2] != (fh, fw):
        print("No usable camera frame, skipping", path)
        continue

    crop = frame[cy1:cy1 + side, cx1:cx1 + side]
    crop = cv2.resize(crop, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), crop)

    if n % 25 == 0:
        print(f"{n}/{len(jobs)} done, {(time.time() - start) / 60:.1f} min elapsed")

print("Finished. Camera images are in the Camera_dataset folder.")
finish()