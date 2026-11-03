import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import json
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
CONF_MIN = 0.60

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


# ---------- Open the camera stream ----------
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Could not open the stream. Check VLC first.")
    raise SystemExit

print("Running. Press q in the video window to quit.")
history = deque(maxlen=SMOOTH)
last_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the stream.")
        break

    h, w = frame.shape[:2]
    side = int(min(h, w) * BOX_FRACTION)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    crop = frame[y1:y1 + side, x1:x1 + side]

    history.append(predict(crop))
    avg = np.mean(history, axis=0)
    best = int(np.argmax(avg))
        label = f"{classes[best]}  {avg[best] * 100:.0f}%"
    if avg[best] < CONF_MIN:
        label = "Unsure / no wafer map"

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
    last_time = now

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Wafer defect classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()