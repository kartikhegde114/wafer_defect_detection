# Live stream and live demo

Back to the [overview](../README.md).

## Board connection

1. Connect the Pico Plus to the PC with a USB-C cable. The board is reachable at `172.32.0.93` (check with `ping 172.32.0.93`).
2. The board's default image already streams the camera over RTSP at `rtsp://172.32.0.93/live/0`. Check it in VLC (Media, then Open Network Stream).
3. The board has no battery clock, so its date resets to 1970 at every power-off, and the burned-in time stamp on the video is wrong. Set it from the PC before a demo:

```
$t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
ssh root@172.32.0.93 "date -s '$t'"
```

## Live classification (`src/live/live_classify2.py`)

For each frame, the script cuts a square from the center (60% of the shorter side), converts BGR to RGB, resizes to 96 x 96, normalizes, runs the model and draws the predicted class and confidence. It averages the last 5 predictions and shows FPS and frame delay. Press `q` to quit. This script still uses the clean-trained model and a fixed centre crop.

**Problem found: lag and freezing.** The board sends about 25 to 30 frames per second, but CPU inference is slower, so OpenCV's queue of unread frames grew and the picture fell behind.
**Fix:** a background thread reads the stream continuously and keeps only the newest frame, the decoder is set to low delay, and the stream reconnects automatically.
**Measured result:** about 20 FPS with roughly 66 ms frame delay on a CPU-only Windows PC.

## Choosing a confidence limit (`src/camera/conf_check.py`)

On the 135 camera validation photos, `model_camera.pth` had an average confidence of 0.81 when right and 0.57 when wrong. Keeping only answers at confidence 0.50 or higher kept 83% of the photos at 85.7% accuracy, so `CONF_MIN = 0.50` was chosen. This was chosen on validation photos, not on test images.

## Live demo on the test images (`src/live/demo_live.py`)

1. It shows a black screen, then a screen with a white square, and finds the square in the camera picture by subtracting the two camera frames. This sets the crop automatically each run.
2. It shows each of the 136 test wafer images on the monitor, one at a time.
3. The Pico Plus camera sees the screen, the calibrated crop is cut out, and `model_camera.pth` predicts. The screen shows the true class, the prediction, the confidence, and CORRECT, WRONG or UNSURE (confidence below 0.50).

**Result of one run (136 test images, camera slightly disturbed during the run):**

| Measure | Value |
|---|---|
| Accuracy if every image gets an answer | 74.3% (101 of 136) |
| Images answered at confidence 0.50 or higher | 118 (87%) |
| Accuracy on the answered images | 79.7% (94 of 118) |
| Marked UNSURE | 18 |

This is close to the 75.7% measured on saved camera photos of the same test images. With 136 images, each figure is uncertain by roughly plus or minus 7 points, and this was a single run, so run-to-run variation is not known. The 136 test images have now been used for several evaluations (clean images, saved camera photos, live demo), although no setting was tuned on them.

## Known issue: the model has no "not a wafer" answer

Pointed at a dark, switched-off monitor, the live window showed "Local 56%". The model must choose one of its 9 classes for any picture.

`src/live/ood_test.py` feeds both models five pictures with no wafer (black, dark grey, mid grey, white, random noise). The clean-trained `model_v2.pth` answered **Local with 57% to 66% confidence on all five**, so "Local" is its fallback for anything unfamiliar. The camera-fine-tuned `model_camera.pth` was much less certain: its top guess was only 15% to 25% in each case, and the top class varied (Local, Scratch, random). A confidence limit therefore behaves differently for the two models: a limit of 0.60 would still let four of the five empty pictures through for `model_v2.pth`, and reject all five for `model_camera.pth`. These are five synthetic pictures, not real photos of an empty screen or room, so the result is only a first indication. A proper fix is a "no wafer" class.

## Limitations

- Small dataset (902 images, 136 for testing), so results have wide uncertainty.
- Very low image resolution (32 x 32); wafer-map pattern images, not real inspection images.
- Camera results come from one camera, one monitor and similar lighting, and the camera moved slightly during capture.
- The live demo was run once, with the camera slightly disturbed, so run-to-run variation is unknown. It reuses the 136 test images shown on a monitor, not new wafers.
- No "no wafer" class.
- One train, validation and test split; no cross-validation.
- The board only streams video; the model does not run on its NPU.

## Planned work

1. Repeat the live demo a few times with a firmly mounted camera to see run-to-run variation, and add a short recorded clip.
2. A second capture session with different lighting and camera positions, as a fresh test set.
3. Test masking the area outside the wafer circle (suggested by the Grad-CAM pictures).
4. Add a "no wafer" class.
5. Optional: convert the model to INT8 with RKNN-Toolkit2 and run it on the board's NPU.

## Scripts

| Script | Purpose |
|---|---|
| `src/live/stream_test.py` | Open the board's RTSP stream in OpenCV |
| `src/live/live_classify.py` | First live classifier (lags) |
| `src/live/live_classify2.py` | Live classifier with newest-frame reader |
| `src/live/demo_live.py` | Live demo: show test images on screen, photograph, classify, score |
| `src/live/ood_test.py` | Test the model on pictures with no wafer |
