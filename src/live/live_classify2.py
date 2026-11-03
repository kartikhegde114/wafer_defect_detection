import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------- Settings ----------
URL = "rtsp://172.32.0.93/live/0"
MODEL_FILE = "model_v2.pth"
IMG = 96
BOX_FRACTION = 0.6
SMOOTH = 5

torch.set_num_threads(min(4, os.cpu_count() or 1))


# ---------- Reads the stream in the background, keeps only the newest frame ----------
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
                print("Cannot open the stream, retrying in 2 seconds...")
                time.sleep(2)
                continue
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    print("Stream lost, reconnecting...")
                    break
                with self.lock:
                    self.frame = frame
                    self.stamp = time.time()
            cap.release()

    def get(self):
        with self.lock:
            return self.frame, self.stamp

    def stop(self):
        self.running = False


# ---------- Load the class names and the trained model ----------
with open("split.json") as f:
    classes = json.load(f)["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()


# ---------- Main loop ----------
grabber = LatestFrame(URL)
history = deque(maxlen=SMOOTH)
last_stamp = 0.0
last_time = time.time()
fps = 0.0

cv2.namedWindow("Wafer defect classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Wafer defect classification", 1280, 720)
print("Running. Press q in the video window to quit.")
while True:
    frame, stamp = grabber.get()
    if frame is None or stamp == last_stamp:
        cv2.waitKey(5)
        continue
    last_stamp = stamp
    frame = frame.copy()

    h, w = frame.shape[:2]
    side = int(min(h, w) * BOX_FRACTION)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    crop = frame[y1:y1 + side, x1:x1 + side]

    history.append(predict(crop))
    avg = np.mean(history, axis=0)
    best = int(np.argmax(avg))
    label = f"{classes[best]}  {avg[best] * 100:.0f}%"

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
    last_time = now
    delay = now - stamp

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"{fps:.1f} FPS  delay {delay * 1000:.0f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Wafer defect classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

grabber.stop()
cv2.destroyAllWindows()