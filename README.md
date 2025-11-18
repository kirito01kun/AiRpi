# Suspicious Behavior Detector (Tiny YOLO + OpenCV)

This project runs a lightweight person detector (YOLOv4-tiny) on a laptop webcam, performs centroid tracking and simple behavior heuristics (sudden acceleration, long stillness, and entering a restricted zone), prints alerts in real time and logs them to `logs/suspicious.log`.

Files created:

- `suspicious_behavior.py` — main script
- `requirements.txt` — Python dependencies
- `models/` — (auto)downloaded model files: `yolov4-tiny.weights`, `yolov4-tiny.cfg`, `coco.names`
- `logs/suspicious.log` — generated alert log

Quick start (Windows):

1. Open PowerShell and change to the desktop (where files are):

```powershell
cd $HOME\Desktop
python -m venv venv
; .\venv\Scripts\Activate.ps1
; python -m pip install --upgrade pip
; pip install -r requirements.txt
```

2. Run the script:

```powershell
python suspicious_behavior.py
```

The script will auto-download `yolov4-tiny.weights` (~6 MB compressed, ~23 MB?) and the cfg/names into `models/` if missing. The first run may take some time.

How it works (summary):

- Loads YOLOv4-tiny using OpenCV DNN (CPU-only) and detects people (COCO class `person`).
- Uses a simple centroid tracker to match detections across frames.
- Computes velocities and acceleration from centroid history.
- Heuristics:
  - Sudden acceleration: acceleration > configured threshold -> alert
  - Long stillness: minimal movement for N seconds -> alert
  - Restricted zone: entering defined rectangular area -> alert

Configuration (in `suspicious_behavior.py`):

- `YOLO_INPUT_SIZE` — change to `320` for faster, lower-accuracy inference (useful for Raspberry Pi).
- `SUDDEN_ACCELERATION_THRESHOLD`, `STILLNESS_TIME_THRESHOLD`, and `STILLNESS_DISTANCE_THRESHOLD` — tune to your camera and scene.
- `RESTRICTED_ZONE` — rectangle fraction (x1,y1,x2,y2) of frame to mark as restricted.

Switching to Raspberry Pi (notes and recommendations):

- Recommended Pi: **Raspberry Pi 4 (4GB/8GB)** or **Pi 5** for reasonable CPU-only performance.
- For real-time detection on Pi, prefer these optimizations:
  - Build OpenCV from source with NEON, VFPv4, and optionally OpenCL enabled, or use prebuilt wheels compatible with your Pi OS.
  - Use `YOLO_INPUT_SIZE = 320` to reduce compute.
  - Increase swap temporarily for compilation: `sudo dphys-swapfile swapoff` -> adjust config -> `sudo dphys-swapfile swapon` (be careful; see Pi docs).
  - Consider hardware accelerators: Coral USB TPU (Edge TPU) or Intel NCS2 (requires model conversion) — these require model conversion to TensorFlow Lite or OpenVINO and code changes.

Pi-specific installation hints (brief):

1. Create virtualenv and install dependencies (prefer system-provided OpenCV or build it):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install numpy first
pip install numpy
# Install OpenCV (prefer prebuilt for your Pi OS or build from source)
pip install opencv-python-headless
pip install requests
```

2. Set `YOLO_INPUT_SIZE = 320` in `suspicious_behavior.py` and run with `python3 suspicious_behavior.py`.

Camera notes (switching to Pi Camera Module):

- The script uses OpenCV `VideoCapture(0)` by default for USB webcams. To use the Pi Camera Module you can:
  - Use `libcamera` with OpenCV (newer Raspberry Pi OS): use `cv2.VideoCapture("/dev/video0")` after enabling `libcamera` and `v4l2` support, or
  - Use the `picamera`/`picamera2` libraries and convert frames to OpenCV arrays. Example (commented snippet in code):

```python
# For Pi Camera v2 + picamera2 (modern):
# from picamera2 import Picamera2
# picam = Picamera2()
# picam.configure(picam.create_preview_configuration(main={'format':'XRGB8888','size':(640,480)}))
# picam.start()
# frame = picam.capture_array()
```

Notes & tuning:

- Threshold constants in `suspicious_behavior.py` are conservative defaults — tune them for your scene and camera height.
- For production usage, add persistence (e.g., avoid repeating same alert), integrate with alerting backends, and add better multi-object tracking.
- If you want to run faster, consider converting the model to a format for hardware accelerators (TFLite for Coral TPU, OpenVINO for Intel NCS, etc.).

If you'd like, I can:

- Run tests or help tune thresholds interactively.
- Add a lightweight web UI to view alerts remotely.
- Add persistent suppression of repeated alerts for the same object.

Enjoy — run `python suspicious_behavior.py` to start.
