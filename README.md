# Wafer-Map Defect Classification with a Live Camera Stream (Luckfox Pico Plus)

This project classifies semiconductor wafer-map defect patterns (Center, Donut, Edge Local, Edge Ring, Local, Scratch, Near full, Random, None) with a ResNet18 transfer-learning model trained in Python.

* On clean images, the model reaches **86.8% accuracy and 0.865 macro F1** on 136 held-out test images.
* Seen through a **Luckfox Pico Plus camera** (images shown on a monitor and photographed), a model trained only on clean images drops to **11.8% accuracy**. After fine-tuning on camera photos, it reaches **75.7% accuracy and 0.747 macro F1** on 136 held-out camera test photos.
* The board streams video over RTSP, and Python with OpenCV reads it at about **20 FPS with roughly 66 ms latency** on a CPU-only Windows PC.

!\[Clean wafer maps, one per class](sample\_images.png)

> \*\*Scope:\*\* the data are 32 x 32 wafer-map pattern images, not SEM or optical inspection images, and the model classifies whole images instead of locating defects. The board acts as the camera, and the model runs on the PC. This is a learning project, not a production inspection system.

\---

## 1\. Why wafer-map patterns matter

In wafer manufacturing, defects form spatial signatures across the wafer: a ring at the edge, a cluster at the center, a scratch line, random scatter. The pattern points toward the process step or equipment issue behind it, so classifying patterns supports root cause analysis and yield learning. This project is a scaled-down version of that idea.

## 2\. Pipeline

```
Luckfox Pico Plus camera --RTSP--> Windows PC (Python + OpenCV) --> ResNet18 classifier --> class label drawn on live video
                                        |
                                        +--> VLC (raw stream view)
```

## 3\. Tools

|Item|Detail|
|-|-|
|Language|Python 3 (virtual environment)|
|Libraries|PyTorch, torchvision, OpenCV, scikit-learn, matplotlib, Pillow, NumPy|
|Model|ResNet18 pretrained on ImageNet, last layer replaced with 9 outputs|
|Hardware|Luckfox Pico Plus (RV1103) with camera, connected to the PC over USB|
|PC|Windows, PowerShell, CPU only|

## 4\. Dataset

* Source: **\[FILL: dataset name, link and license]**. The images are not included in this repository.
* **902 images**, 9 classes, about 100 per class, all **32 x 32 pixels, RGB**.
* Folder layout expected: `Wafer\_dataset/<class name>/<image files>`, with no ready-made train/validation/test split.

|Class|Images|
|-|-|
|Center|100|
|Donut|102|
|Edge Local|103|
|Edge Ring|102|
|Local|100|
|near full|95|
|none|100|
|random|100|
|Scratch|100|

The classes are nearly balanced, so no class-imbalance handling was needed.

## 5\. Method on clean images

### Split

Images are divided into **631 train / 135 validation / 136 test** (70/15/15), stratified so each part keeps the same share of every class (random seed 42). The exact indices are saved in `split.json`, so every script uses the same split.

### Model 1 (`train.py`)

Images resized to 64 x 64 and normalized for the pretrained model, with random flips and rotation on training images. ResNet18 with a new 9-output layer, Adam optimizer (learning rate 0.001), 15 epochs, batch size 32. Validation accuracy peaked at 86.7%.

### Model 2 (`train2.py`)

Same split, with three changes: input size 96 x 96, AdamW with learning rate 0.0003 and a cosine schedule, and label smoothing 0.1, for 30 epochs. Validation accuracy peaked at 90.4% and was much steadier between epochs.

The best epoch is always chosen on the validation images only. The test images are used once per model, for final reporting.

## 6\. Results on clean images (136 test images)

|Metric|Model 1 (64 px)|Model 2 (96 px)|
|-|-|-|
|Accuracy|77.2% (105 of 136)|**86.8%** (118 of 136)|
|Macro F1|0.773|**0.865**|

|Class|Precision|Recall|F1 (Model 2)|F1 (Model 1)|
|-|-|-|-|-|
|Center|0.929|0.867|0.897|0.769|
|Donut|0.875|0.875|0.875|0.839|
|Edge Local|0.933|0.933|0.933|0.625|
|Edge Ring|0.938|0.938|0.938|0.882|
|Local|0.692|0.600|0.643|0.621|
|Scratch|0.812|0.867|0.839|0.593|
|near full|0.875|1.000|0.933|0.923|
|none|0.938|1.000|0.968|0.966|
|random|0.786|0.733|0.759|0.737|

!\[Confusion matrix of Model 2 on clean test images](confusion\_matrix\_v2.png)

**Analysis**

* The largest gains came on Edge Local (F1 0.625 to 0.933) and Scratch (0.593 to 0.839), the two classes Model 1 confused most often.
* For Model 2 the weakest classes are **Local (0.643) and random (0.759)**.
* Three things changed between the models (resolution, learning-rate schedule, label smoothing), so this experiment cannot say which one caused the gain.
* With only 136 test images, one accuracy figure is uncertain by roughly plus or minus 6 points.
* The same test set was used for both models. No tuning was done against it, but it has been seen twice.

\---

## 7\. From clean images to camera photos

### 7.1 Why this step

A model trained on clean digital images may fail on pictures taken by a camera: blur, different colours, glare and framing all change. To measure this, every dataset image was shown on a monitor and photographed with the Pico Plus camera.

### 7.2 Capture (`capture\_camera.py`)

* The script shows each of the 902 images full screen, waits for the camera to settle, and saves the newest frame.
* **Calibration:** before capturing, the script shows a white square on a black screen, finds it in the camera picture, and crops exactly that region (128 x 128 pixels saved). The crop box is written to `crop\_box.json`, and `calibration\_check.jpg` shows it drawn on the camera view.
* Images are shown in random order, so lighting drift does not line up with one class.
* The photos keep the same split as the clean images: 631 train / 135 validation / 136 test. The test photos show images the model never trained on.

!\[Camera view with the calibrated crop box](calibration\_check.jpg)

!\[Camera photos of the displayed wafer maps, three per class](camera\_samples.png)

### 7.3 Problems found along the way

1. **First capture was invalid.** Average brightness was about 120 for every class and the photos differed from the overall average by only 1.0 out of 255, so nearly all photos were the same picture: the camera was not looking at the wafer. A self-test was added, and the capture was repeated.
2. **Second capture had inconsistent framing.** The wafer sat off-centre or was cut off in many photos, because the crop was a fixed square in the middle of the camera picture and the camera view did not line up with the screen. The calibration step above fixed this.
3. **The camera still moved slightly during the final capture**, so wafer position varies a little between photos. Training therefore uses random shifts and zoom.
4. The photos are blurrier and more washed-out blue than the clean images.

Results from the two rejected captures are not reported.

### 7.4 Fine-tuning (`train\_camera.py`)

Starts from the clean-trained `model\_v2.pth` and trains on the 631 camera training photos (25 epochs, AdamW, learning rate 0.0001, cosine schedule, label smoothing 0.1). Training photos get random flips, small rotation, shifts up to 12%, zoom from 85% to 115%, brightness, contrast and colour changes, and occasional blur. The best epoch is chosen on the 135 camera validation photos (78.5% accuracy), and the model is then tested once on the 136 camera test photos.

### 7.5 Results on camera test photos (136 photos)

|Model|Accuracy|Macro F1|
|-|-|-|
|Trained on clean images only (`model\_v2.pth`)|11.8%|0.037|
|Fine-tuned on camera photos (`model\_camera.pth`)|**75.7%**|**0.747**|

Before fine-tuning, the model labelled almost every camera photo as "Local" (132 of 136), which is close to chance level (11.1% for 9 classes).

!\[Clean-trained model on camera test photos](confusion\_matrix\_camera\_before.png)

!\[Fine-tuned model on camera test photos](confusion\_matrix\_camera\_after.png)

|Class|Precision|Recall|F1|
|-|-|-|-|
|Center|0.714|0.667|0.690|
|Donut|0.824|0.875|0.848|
|Edge Local|0.588|0.667|0.625|
|Edge Ring|0.867|0.812|0.839|
|Local|0.778|0.467|0.583|
|Scratch|0.636|0.467|0.538|
|near full|0.875|1.000|0.933|
|none|0.750|1.000|0.857|
|random|0.765|0.867|0.812|

**Analysis**

* Fine-tuning moved the model from nearly always answering "Local" to a clear diagonal in the confusion matrix. Camera test accuracy (75.7%) is about 11 points below the clean-image result (86.8%), so a gap remains.
* Strongest classes: near full, Donut, Edge Ring, none and random. Weakest: **Scratch (F1 0.538) and Local (0.583)**. Six of the 15 Scratch photos were labelled Edge Local, the same confusion seen on clean images, probably because a streak near the edge looks like an edge cluster at this resolution and blur.
* Center (F1 0.690) and Edge Local (0.625) are right about two thirds of the time.
* With 136 test photos, one accuracy figure is uncertain by roughly plus or minus 7 points.
* These results hold for this camera, this monitor and this lighting. Other cameras or screens were not tested.

\---

## 8\. Live camera pipeline

### Board connection

1. Connect the Pico Plus to the PC with a USB-C cable. The board is reachable at `172.32.0.93` (check with `ping 172.32.0.93`).
2. The board's default image already streams the camera over RTSP at `rtsp://172.32.0.93/live/0`. Check it in VLC (Media > Open Network Stream).
3. The board has no battery clock, so its date resets to 1970 at every power-off, and the burned-in time stamp on the video is wrong. Set it from the PC before a demo:

```
$t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
ssh root@172.32.0.93 "date -s '$t'"
```

### Live classification (`live\_classify2.py`)

For each frame, the script cuts a square from the center (60% of the shorter side), converts BGR to RGB, resizes to 96 x 96, normalizes, runs the model and draws the predicted class and confidence. It averages the last 5 predictions and shows FPS and frame delay. Press `q` to quit.

**Problem found: lag and freezing.** The board sends about 25 to 30 frames per second, but CPU inference is slower, so OpenCV's queue of unread frames grew and the picture fell behind.
**Fix:** a background thread reads the stream continuously and keeps only the newest frame, the decoder is set to low delay, and the stream reconnects automatically.
**Measured result:** about 20 FPS with roughly 66 ms frame delay on a CPU-only Windows PC.

**Current status:** the live script still loads the clean-trained `model\_v2.pth` and uses a fixed centre crop. It has **not yet been updated** to use `model\_camera.pth` and the calibrated crop from `crop\_box.json`, so the camera accuracy in section 7 is measured on saved camera photos, not yet on the live window.

### Known issue: the model has no "not a wafer" answer

Pointed at a dark, switched-off monitor, the live window showed "Local 56%". The model must choose one of its 9 classes for any picture. ood\_test.py feeds both models five pictures with no wafer (black, dark grey, mid grey, white, random noise). The clean-trained model\_v2.pth answered Local with 57% to 66% confidence on all five, so “Local” is its fallback for anything unfamiliar. The camera-fine-tuned model\_camera.pth was much less certain: its top guess was only 15% to 25% in each case, and the top class varied (Local, Scratch, random). A confidence limit therefore behaves differently for the two models: a limit of 0.60 would still let four of the five empty pictures through for model\_v2.pth, and reject all five for model\_camera.pth. These are five synthetic pictures, not real photos of an empty screen or room, so the result is only a first indication. The right limit would have to be chosen on real camera photos, and a proper fix is a “no wafer” class.

## 9\. Limitations

* Small dataset (902 images, 136 for testing), so results have wide uncertainty.
* Very low image resolution (32 x 32).
* Wafer-map pattern images, not real inspection images.
* Camera results come from one camera, one monitor and similar lighting, and the camera moved slightly during capture.
* Camera accuracy was measured on saved photos, not yet on the live window.
* No "no wafer" class.
* One train/validation/test split; no cross-validation.
* The board only streams video; the model does not run on its NPU.

## 10\. Planned work

1. Update the live script to use `model\_camera.pth` and the calibrated crop, and record a short demo.
2. A second capture session with different lighting and camera positions, to test whether more variety improves the weak classes (Scratch, Local, Edge Local).
3. Add a "no wafer" class.
4. Optional: convert the model to INT8 with RKNN-Toolkit2 and run it on the board's NPU.

## 11\. Repository structure

```
check\_dataset.py       count images per class
inspect\_images.py      check image size and colour, make sample\_images.png
train.py               first training run (64 px), writes split.json and model\_best.pth
evaluate.py            first test-set evaluation
train2.py              second training run (96 px), writes model\_v2.pth
evaluate2.py           second test-set evaluation, writes confusion\_matrix\_v2.png
capture\_camera.py      show each dataset image on screen and photograph it with the board camera
check\_camera\_photos.py brightness statistics and a picture grid of the camera photos
eval\_camera.py         test a model on the camera test photos
train\_camera.py        fine-tune on camera photos, before/after evaluation
stream\_test.py         open the board's RTSP stream in OpenCV
live\_classify.py       first live classifier (lags)
live\_classify2.py      live classifier with newest-frame reader
ood\_test.py            test the model on pictures with no wafer
split.json             saved train/validation/test indices
requirements.txt
```

The dataset, the camera photos, model files (`\*.pth`) and the virtual environment are not stored in the repository.

## 12\. How to run

```
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# put the dataset in Wafer\_dataset\\<class name>\\ first
python check\_dataset.py
python train.py
python train2.py
python evaluate2.py

# with the Pico Plus connected, streaming, and mounted facing the monitor:
python capture\_camera.py
python eval\_camera.py
python train\_camera.py

python live\_classify2.py
```

`requirements.txt`:

```
torch
torchvision
opencv-python
scikit-learn
matplotlib
Pillow
numpy
```

## 13\. Skills demonstrated

Dataset inspection, stratified splitting, CNN transfer learning, data augmentation, model evaluation (accuracy, macro F1, per-class metrics, confusion matrix), error analysis, measuring and closing a domain gap (clean images vs camera photos), automatic calibration, RTSP video streaming, real-time inference with OpenCV, debugging stream latency, honest reporting of limitations.

