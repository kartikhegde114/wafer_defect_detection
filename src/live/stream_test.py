import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2

URL = "rtsp://172.32.0.93/live/0"
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Could not open the stream. Check VLC first.")
    raise SystemExit

print("Stream opened. Press q in the video window to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the stream.")
        break
    cv2.imshow("Pico Plus camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()