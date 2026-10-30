import os
# Receive the video over TCP with low delay, like in the capture script
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

import ctypes
import json
import random
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms, models

# ---------- Settings ----------
URL = "rtsp://172.32.0.93/live/0"     # the board's camera stream
MODEL_FILE = "model_camera.pth"       # the model fine-tuned on camera photos
IMG = 96                              # input size the model was trained on
RATIO = 0.6                           # wafer size on screen, as a share of the screen height
SETTLE = 0.7                          # seconds to wait so the camera adjusts to a new image
HOLD = 1.0                            # seconds to show each result
CONF_MIN = 0.50                       # below this confidence the answer is "UNSURE"
WIN = "Live wafer demo"

# Read the real screen size on Windows so the picture fills the screen
ctypes.windll.user32.SetProcessDPIAware()
SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)


# ---------- Reads the stream in the background and keeps only the newest frame ----------
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
                time.sleep(2)             # stream not ready, try again in 2 seconds
                continue
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break                 # stream lost, reconnect
                with self.lock:
                    self.frame = frame
                    self.stamp = time.time()
            cap.release()

    def get(self):
        with self.lock:
            return self.frame, self.stamp


# ---------- Load the class names and the trained model ----------
with open("split.json") as f:
    split = json.load(f)
classes = split["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
model.eval()                              # prediction mode, no learning

# Same preparation as for the camera photos in training: resize and normalize
tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(crop_bgr):
    """Return the 9 class probabilities for one camera crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)          # OpenCV is BGR, the model wants RGB
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)  # same size as the saved training photos
    x = tf(Image.fromarray(small)).unsqueeze(0)              # add a batch dimension
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()


def make_canvas(wafer_bgr, lines=None, crop_bgr=None):
    """Grey screen with the wafer in the middle and optional result text around it."""
    canvas = np.full((SCREEN_H, SCREEN_W, 3), 60, dtype=np.uint8)
    size = int(min(SCREEN_H, SCREEN_W) * RATIO)
    big = cv2.resize(wafer_bgr, (size, size), interpolation=cv2.INTER_NEAREST)  # keep the blocks sharp
    top = (SCREEN_H - size) // 2
    left = (SCREEN_W - size) // 2
    canvas[top:top + size, left:left + size] = big
    if lines:
        for i, (text, colour) in enumerate(lines):           # result text in the top-left corner
            cv2.putText(canvas, text, (30, 60 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2)
    if crop_bgr is not None:                                 # what the model saw, bottom-left
        thumb = cv2.resize(crop_bgr, (256, 256))
        canvas[SCREEN_H - 296:SCREEN_H - 40, 30:286] = thumb
        cv2.putText(canvas, "Camera crop", (30, SCREEN_H - 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def wait(seconds):
    """Keep the window alive for a while. Returns False if q was pressed."""
    end = time.time() + seconds
    while time.time() < end:
        if cv2.waitKey(10) & 0xFF == ord("q"):
            return False
    return True


def quit_now():
    cv2.destroyAllWindows()
    raise SystemExit


# ---------- Start the camera and the full-screen window ----------
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
grabber = LatestFrame(URL)

# ---------- Calibration: find the wafer area by comparing a black screen with a white square ----------
size = int(min(SCREEN_H, SCREEN_W) * RATIO)
top = (SCREEN_H - size) // 2
left = (SCREEN_W - size) // 2

black = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)    # all-black screen
square = black.copy()
square[top:top + size, left:left + size] = 255               # the same screen with a white square


def grab_after(canvas, seconds):
    """Show a screen, wait so the camera can adapt, and return the newest camera frame."""
    cv2.imshow(WIN, canvas)
    wait(seconds)
    f, _ = grabber.get()
    return f


# Wait up to 15 seconds for the first camera frame (opening the stream can take a few seconds)
cv2.imshow(WIN, black)
frame = None
start_wait = time.time()
while frame is None and time.time() - start_wait < 15:
    cv2.waitKey(100)
    frame, _ = grabber.get()
if frame is None:
    print("No camera frame after 15 seconds. Check the stream in VLC.")
    quit_now()

frame_black = grab_after(black, 3.0)                         # picture with the black screen
frame_square = grab_after(square, 3.0)                       # picture with the white square
cv2.imwrite("calibration_black.jpg", frame_black)            # saved so we can check what the camera saw
cv2.imwrite("calibration_debug.jpg", frame_square)

fh, fw = frame_square.shape[:2]
if frame_black.shape[:2] != (fh, fw):
    print("The camera picture changed size during calibration. Run it again.")
    quit_now()


def to_gray(img):
    """Blurred grey picture as whole numbers, so subtracting two pictures works."""
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.int16)


# Keep only the pixels that became BRIGHTER when the white square appeared
diff = np.clip(to_gray(frame_square) - to_gray(frame_black), 0, 255).astype(np.uint8)
diff[0:int(fh * 0.13), 0:int(fw * 0.25)] = 0                 # ignore the time stamp, its digits change every second
_, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)    # pixels that brightened by more than 40
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # fill small gaps
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("The white square did not show up in the camera picture. See calibration_debug.jpg.")
    quit_now()

bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
brighter = float(diff[by:by + bh, bx:bx + bw].mean())        # how much brighter the square area became
aspect = bw / max(bh, 1)
area_share = (bw * bh) / (fw * fh)
touches_edge = bx <= 2 or by <= 2 or bx + bw >= fw - 2 or by + bh >= fh - 2
print(f"Square found: x={bx} y={by} w={bw} h={bh} in a {fw}x{fh} picture, brighter by {brighter:.0f}")

side = int(min(bw, bh) * 0.96)                               # crop slightly inside the bright area
cx1 = int(np.clip(bx + bw // 2 - side // 2, 0, fw - side))
cy1 = int(np.clip(by + bh // 2 - side // 2, 0, fh - side))

# Save the check picture BEFORE the checks, so it exists even when calibration is rejected
check = frame_square.copy()
cv2.rectangle(check, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)          # red: the bright area that was found
cv2.rectangle(check, (cx1, cy1), (cx1 + side, cy1 + side), (0, 255, 0), 4)  # green: the crop that will be used
cv2.imwrite("calibration_demo_check.jpg", check)

# Checks: clearly brighter, roughly square (now allowing a slight angle), sensible size, not cut off
if brighter < 40 or not (0.6 <= aspect <= 1.7) or not (0.04 <= area_share <= 0.9) or touches_edge:
    print("The square is not clearly and completely visible. See calibration_demo_check.jpg, then move the camera and try again.")
    quit_now()
print("Calibration done. See calibration_demo_check.jpg")

# ---------- The wafer images to show: the 136 TEST images, in random order ----------
data = datasets.ImageFolder("Wafer_dataset")
jobs = [data.samples[i] for i in split["test"]]              # (path, label) pairs
random.seed(1)
random.shuffle(jobs)

shown = forced_correct = answered = answered_correct = unsure = 0

# ---------- Main loop: show, photograph, predict, display the result ----------
for path, label in jobs:
    wafer = cv2.imread(path)
    cv2.imshow(WIN, make_canvas(wafer))                      # wafer only, no text
    t_show = time.time()
    if not wait(SETTLE):
        break

    frame, stamp = grabber.get()                             # newest frame after the screen changed
    waited = time.time()
    while (frame is None or stamp < t_show + SETTLE) and time.time() - waited < 5:
        cv2.waitKey(10)
        frame, stamp = grabber.get()
    if frame is None or frame.shape[:2] != (fh, fw):
        continue

    crop = frame[cy1:cy1 + side, cx1:cx1 + side]             # the calibrated wafer area
    p = predict(crop)
    best = int(p.argmax())
    conf = float(p[best])

    shown += 1
    forced_correct += int(best == label)                     # accuracy if we always had to answer
    if conf < CONF_MIN:
        verdict, colour = "UNSURE", (0, 200, 255)
        unsure += 1
    else:
        answered += 1
        ok = best == label
        answered_correct += int(ok)
        verdict, colour = ("CORRECT", (0, 220, 0)) if ok else ("WRONG", (0, 0, 255))

    lines = [
        (f"True: {classes[label]}", (255, 255, 255)),
        (f"Predicted: {classes[best]} ({conf * 100:.0f}%)", (255, 255, 255)),
        (verdict, colour),
        (f"Shown {shown}   correct {answered_correct}   unsure {unsure}", (200, 200, 200)),
    ]
    cv2.imshow(WIN, make_canvas(wafer, lines, crop))
    if not wait(HOLD):
        break

# ---------- Summary ----------
print("\n--- Live demo summary ---")
print("Images shown:", shown)
if shown:
    forced_pct = forced_correct / shown * 100       # accuracy if every image gets an answer
    answered_pct = answered / shown * 100           # share of images that passed the confidence limit
    print("Accuracy if every image gets an answer:", round(forced_pct, 1), "%")
    print("Images answered at confidence >=", CONF_MIN, ":", answered, "(", round(answered_pct), "% )")
    if answered:
        print("Accuracy on the answered images:", round(answered_correct / answered * 100, 1), "%")
    print("Unsure:", unsure)
quit_now()