#!/usr/bin/env python3
"""
suspicious_behavior.py

Real-time person detection + simple behavior heuristics using YOLOv4-tiny and OpenCV DNN.

Features:
- Auto-downloads `yolov4-tiny.weights`, `yolov4-tiny.cfg`, and `coco.names` into `models/` if missing.
- Uses laptop webcam (index 0) by default.
- Detects people using Tiny YOLO and does centroid tracking.
- Heuristics: sudden acceleration, long stillness, entering restricted zone.
- Prints alerts, overlays boxes/IDs on video, saves logs to `logs/suspicious.log`.

Optimized for later Raspberry Pi use (CPU-only, lightweight operations, configurable input size).

Run: python suspicious_behavior.py

Author: Generated for user
"""
import os
import sys
import time
import math
import argparse
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
import json

# ------------------------ Configuration ------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Download sources (real links)
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"

COCO_NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")
YOLO_CFG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny.weights")

# Tiny YOLO input size (tradeoff speed vs accuracy). For Raspberry Pi you may reduce to 320.
YOLO_INPUT_SIZE = 416

# Detection parameters
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

# Tracking / heuristic parameters (adjust to environment)
MAX_DISTANCE = 80  # px - max centroid distance to consider same object
SUDDEN_ACCELERATION_THRESHOLD = 3000.0  # px/s^2 - tune for your camera (raised default to reduce sensitivity)
STILLNESS_TIME_THRESHOLD = 8.0  # seconds of near-zero movement -> loitering/collapse
STILLNESS_DISTANCE_THRESHOLD = 10.0  # px - movement below this considered still

# Fall detection parameters
FALL_HEIGHT_RATIO = 0.6  # if height drops below this ratio vs ~1s earlier -> candidate
FALL_DESCENT_SPEED = 200.0  # px/s downward speed threshold to consider falling
FALL_STILLNESS_TIME = 3.0  # seconds of stillness after fall to confirm
ASPECT_RATIO_THRESHOLD = 1.2  # w/h greater than this suggests horizontal orientation

# Overlay UI: duration to show top-left action box (seconds)
OVERLAY_DURATION = 5.0
# Restricted zone: rectangle defined as (x1,y1,x2,y2) in relative coords (fractions)
# Example: bottom center area. Values are fractions of frame width/height
RESTRICTED_ZONE = (0.3, 0.6, 0.7, 0.95)
RESTRICTED_ZONE_FILE = os.path.join(MODEL_DIR, "restricted_zone.json")

LOG_FILE = os.path.join(LOG_DIR, "suspicious.log")

# Logging suppression: avoid writing identical alerts repeatedly in a short time window.
# This helps keep `logs/suspicious.log` from being flooded with the same message each frame.
LOG_COOLDOWN_SECONDS = 5.0
# Internal state for logging suppression
_last_log_text = None
_last_log_time = 0.0
_last_log_times = {}

# ------------------------ Utilities ------------------------
def download_file(url, dest_path, chunk_size=8192):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Found existing {os.path.basename(dest_path)} — skipping download")
        return
    print(f"Downloading {url} -> {dest_path}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r{pct:3.0f}% ", end='', flush=True)
    print("\nDownload completed")

def ensure_models():
    # Try to download model files if missing
    try:
        if not os.path.exists(COCO_NAMES_PATH):
            download_file(COCO_NAMES_URL, COCO_NAMES_PATH)
        if not os.path.exists(YOLO_CFG_PATH):
            download_file(YOLO_CFG_URL, YOLO_CFG_PATH)
        if not os.path.exists(YOLO_WEIGHTS_PATH):
            download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    except Exception as e:
        print("Error downloading models:", e)
        print("Please check your internet connection or download the files manually into 'models/'")
        raise

def log_alert(text):
    """Append an alert to the log file, but suppress immediate duplicates.

    Suppression rules:
    - If the exact same `text` was logged within `LOG_COOLDOWN_SECONDS`, skip writing again.
    - Also avoid writing the exact same text twice in a row even if timestamps are slightly different.
    """
    global _last_log_text, _last_log_time, _last_log_times
    now_ts = time.time()
    key = text.strip()

    # If identical to last written and within cooldown, skip
    if _last_log_text == key and (now_ts - _last_log_time) < LOG_COOLDOWN_SECONDS:
        return

    # If this specific text was logged recently, skip
    last_time_for_key = _last_log_times.get(key, 0.0)
    if (now_ts - last_time_for_key) < LOG_COOLDOWN_SECONDS:
        return

    # Record and write
    _last_log_text = key
    _last_log_time = now_ts
    _last_log_times[key] = now_ts

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {text}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
    print(line.strip())

# ------------------------ Simple Centroid Tracker ------------------------
class CentroidTracker:
    def __init__(self, max_distance=MAX_DISTANCE):
        self.next_object_id = 1
        self.objects = {}  # id -> centroid
        # id -> deque of (timestamp, centroid, size) where size is (w,h)
        self.history = {}
        self.max_distance = max_distance

    def update(self, input_centroids, input_sizes=None):
        """Match input_centroids to existing objects; return dict id->centroid"""
        now = time.time()
        if len(self.objects) == 0:
            for idx, c in enumerate(input_centroids):
                oid = self.next_object_id
                self.objects[oid] = c
                size = None
                if input_sizes is not None and idx < len(input_sizes):
                    size = input_sizes[idx]
                self.history[oid] = deque([(now, c, size)], maxlen=32)
                self.next_object_id += 1
            return self.objects

        # Build distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[i] for i in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        assigned_objects = set()
        assigned_inputs = set()
        # Greedy matching: pick smallest distance pairs first
        pairs = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                pairs.append((D[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        for dist, i, j in pairs:
            if i in assigned_objects or j in assigned_inputs:
                continue
            if dist > self.max_distance:
                continue
            oid = object_ids[i]
            self.objects[oid] = input_centroids[j]
            size = None
            if input_sizes is not None and j < len(input_sizes):
                size = input_sizes[j]
            self.history[oid].append((now, input_centroids[j], size))
            assigned_objects.add(i)
            assigned_inputs.add(j)

        # Unassigned inputs -> new objects
        for j, ic in enumerate(input_centroids):
            if j in assigned_inputs:
                continue
            oid = self.next_object_id
            self.objects[oid] = ic
            size = None
            if input_sizes is not None and j < len(input_sizes):
                size = input_sizes[j]
            self.history[oid] = deque([(now, ic, size)], maxlen=32)
            self.next_object_id += 1

        # Note: we do not remove disappeared objects in this simple implementation
        return self.objects

    def get_velocity(self, oid, samples=3):
        """Return approximate velocity (vx, vy) in px/sec for object id using last samples."""
        hist = list(self.history.get(oid, []))
        if len(hist) < 2:
            return (0.0, 0.0)
        pts = hist[-samples:]
        if len(pts) < 2:
            return (0.0, 0.0)
        (t0, p0, _s0) = pts[0]
        (t1, p1, _s1) = pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        return (vx, vy)

    def get_speed(self, oid, samples=3):
        vx, vy = self.get_velocity(oid, samples=samples)
        return math.hypot(vx, vy)

    def get_acceleration(self, oid, samples=4):
        """Return approximate acceleration magnitude (px/s^2).

        This computes velocities between successive position samples, then
        computes accelerations between successive velocity samples. To
        reduce spurious alerts from single-frame noise, we return the
        average of the last few acceleration samples (configurable via
        the `samples` parameter).
        """
        hist = list(self.history.get(oid, []))
        if len(hist) < 3:
            return 0.0

        # compute velocities between successive pairs
        velocities = []
        for i in range(1, len(hist)):
            t0, p0, _s0 = hist[i-1]
            t1, p1, _s1 = hist[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            vx = (p1[0] - p0[0]) / dt
            vy = (p1[1] - p0[1]) / dt
            velocities.append((t1, vx, vy))

        if len(velocities) < 2:
            return 0.0

        # compute accelerations between consecutive velocity samples
        accs = []
        for i in range(1, len(velocities)):
            t_prev, vx_prev, vy_prev = velocities[i-1]
            t_last, vx_last, vy_last = velocities[i]
            dt = t_last - t_prev
            if dt <= 0:
                continue
            ax = (vx_last - vx_prev) / dt
            ay = (vy_last - vy_prev) / dt
            accs.append(math.hypot(ax, ay))

        if not accs:
            return 0.0

        # average the last N acceleration samples to smooth spikes
        N = min(len(accs), max(1, samples - 1))
        return sum(accs[-N:]) / N

    def time_still(self, oid, time_window=STILLNESS_TIME_THRESHOLD, dist_threshold=STILLNESS_DISTANCE_THRESHOLD):
        now = time.time()
        hist = list(self.history.get(oid, []))
        if not hist:
            return 0.0
        # compute how long the object has been within dist_threshold
        # walk backwards until movement > dist_threshold
        count_time = 0.0
        for i in range(len(hist)-1, 0, -1):
            t_curr, p_curr, _s_curr = hist[i]
            t_prev, p_prev, _s_prev = hist[i-1]
            dist = math.hypot(p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
            dt = t_curr - t_prev
            if dist > dist_threshold:
                break
            count_time += dt
            if count_time >= time_window:
                break
        return count_time

    def get_height_ratio_change(self, oid, seconds=1.0):
        """Return latest_height / earlier_height where earlier is ~seconds before now. If not available return 1.0"""
        hist = list(self.history.get(oid, []))
        if len(hist) < 2:
            return 1.0
        now = time.time()
        latest_t, _latest_p, latest_s = hist[-1]
        # find earliest entry older than now - seconds
        target_t = latest_t - seconds
        earlier_s = None
        for (t, p, s) in reversed(hist):
            if t <= target_t:
                earlier_s = s
                break
        if earlier_s is None:
            # fallback to first sample
            earlier_s = hist[0][2]
        if earlier_s is None or latest_s is None:
            return 1.0
        # sizes stored as (w,h)
        h_latest = latest_s[1] if isinstance(latest_s, (list, tuple)) and len(latest_s) >= 2 else latest_s
        h_earlier = earlier_s[1] if isinstance(earlier_s, (list, tuple)) and len(earlier_s) >= 2 else earlier_s
        try:
            if h_earlier <= 0:
                return 1.0
            return float(h_latest) / float(h_earlier)
        except Exception:
            return 1.0

# ------------------------ YOLO Helpers ------------------------
def load_yolo(net_cfg, net_weights):
    net = cv2.dnn.readNetFromDarknet(net_cfg, net_weights)
    # Force CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass
    return net

def save_restricted_zone(zone):
    """Save restricted zone to RESTRICTED_ZONE_FILE as JSON.

    zone: dict with keys 'type' and 'coords'
    type == 'rect' -> coords [x1, y1, x2, y2] (relative)
    type == 'poly' -> coords [[x_rel,y_rel], ...]
    """
    try:
        with open(RESTRICTED_ZONE_FILE, 'w', encoding='utf-8') as f:
            json.dump(zone, f)
        print(f"Saved restricted zone -> {RESTRICTED_ZONE_FILE}")
    except Exception as e:
        print("Failed to save restricted zone:", e)

def load_restricted_zone():
    """Load restricted zone from file, return None if not found/invalid."""
    if not os.path.exists(RESTRICTED_ZONE_FILE):
        return None
    try:
        with open(RESTRICTED_ZONE_FILE, 'r', encoding='utf-8') as f:
            zone = json.load(f)
        # basic validation
        if not isinstance(zone, dict) or 'type' not in zone or 'coords' not in zone:
            return None
        return zone
    except Exception as e:
        print("Failed to load restricted zone:", e)
        return None

def interactive_select_zone(orig_frame, min_points=3):
    """Allow the user to click points on `orig_frame` to define a polygon.

    - Left click to add a point
    - Right click to remove the last point
    - 'r' to reset, 's' to save (requires at least `min_points`), 'q' to cancel

    Returns zone dict with type 'poly' and relative coords, or None if cancelled.
    """
    frame = orig_frame.copy()
    overlay = frame.copy()
    points = []
    window_name = 'Select Restricted Zone - click to add points, r=reset, s=save, q=cancel'

    def redraw():
        nonlocal overlay
        overlay = frame.copy()
        # draw existing points and lines
        for i, (x, y) in enumerate(points):
            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(overlay, points[i-1], points[i], (0, 255, 0), 2)
        # if more than two points, show closing line to first
        if len(points) > 2:
            cv2.line(overlay, points[-1], points[0], (0, 200, 0), 1)
        cv2.imshow(window_name, overlay)

    def mouse_cb(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, mouse_cb)

    print(f"Interactive selection: click points on the frame (min {min_points}).")
    print("Controls: left-click=add, right-click=undo, r=reset, s=save, q=cancel")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            # reset
            points = []
            redraw()
            print("Reset points")
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            print("Cancelled zone selection")
            return None
        elif key == ord('s'):
            if len(points) < min_points:
                print(f"Need at least {min_points} points to save, currently have {len(points)}")
                continue
            # compute relative coords
            H, W = orig_frame.shape[:2]
            rel = [[float(x)/W, float(y)/H] for (x, y) in points]
            zone = {'type': 'poly', 'coords': rel}
            cv2.destroyWindow(window_name)
            print(f"Saved polygon with {len(points)} points")
            return zone
        else:
            # ignore other keys
            continue

def get_output_layer_names(net):
    layer_names = net.getLayerNames()
    # OpenCV's `getUnconnectedOutLayers()` can return different shapes/types
    # depending on the OpenCV version (e.g., array of shape (N,1), 1D array, or list of ints).
    try:
        outs = net.getUnconnectedOutLayers()
        # Try to flatten to a simple list of ints
        if hasattr(outs, 'flatten'):
            idxs = outs.flatten().tolist()
        else:
            idxs = list(outs)

        cleaned = []
        for i in idxs:
            # each i might be a scalar int, numpy scalar, or array-like [x]
            if isinstance(i, (list, tuple, np.ndarray)):
                # take first element
                ii = int(np.array(i).flatten()[0])
            else:
                ii = int(i)
            cleaned.append(ii)

        out_layers = [layer_names[i - 1] for i in cleaned]
        return out_layers
    except Exception:
        # Fallback: newer OpenCV provides getUnconnectedOutLayersNames()
        try:
            return net.getUnconnectedOutLayersNames()
        except Exception:
            # As a last resort, return all layer names (less efficient but safe)
            return layer_names

# ------------------------ Main logic ------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Suspicious behavior detection using Tiny YOLO')
    p.add_argument('--camera', type=int, default=0, help='camera index (default 0)')
    p.add_argument('--input', type=str, default=None, help='optional video file')
    p.add_argument('--set-zone', action='store_true', help='Interactively set restricted zone on startup')
    p.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='confidence threshold')
    p.add_argument('--nms', type=float, default=NMS_THRESHOLD, help='NMS threshold')
    p.add_argument('--size', type=int, default=YOLO_INPUT_SIZE, help='YOLO input size (e.g., 416 or 320)')
    p.add_argument('--debug', action='store_true', help='Show per-object debug values on screen')
    p.add_argument('--acc-threshold', type=float, default=SUDDEN_ACCELERATION_THRESHOLD, help='sudden acceleration threshold (px/s^2)')
    return p.parse_args()

def main():
    args = parse_args()
    # Ensure models are present
    ensure_models()

    # Load class names
    with open(COCO_NAMES_PATH, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    person_class_id = classes.index('person') if 'person' in classes else 0

    net = load_yolo(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    output_layer_names = get_output_layer_names(net)

    # tracker
    tracker = CentroidTracker(max_distance=MAX_DISTANCE)

    # Video capture
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Unable to open video source')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input FPS (approx): {fps}")

    # Pre-calc restricted zone in pixels after reading first frame size
    ret, frame = cap.read()
    if not ret:
        print('Unable to read from camera')
        sys.exit(1)
    H, W = frame.shape[:2]

    # Load previously saved restricted zone (if any)
    zone = load_restricted_zone()
    if args.set_zone or zone is None:
        print('No saved restricted zone found or --set-zone requested. Enter interactive selection.')
        sel = interactive_select_zone(frame)
        if sel is not None:
            zone = sel
            save_restricted_zone(zone)
        else:
            # fallback to default rectangle
            zone = {'type': 'rect', 'coords': list(RESTRICTED_ZONE)}

    # If zone is rect, compute pixel bounds now
    if zone.get('type') == 'rect':
        rx1 = int(zone['coords'][0] * W)
        ry1 = int(zone['coords'][1] * H)
        rx2 = int(zone['coords'][2] * W)
        ry2 = int(zone['coords'][3] * H)
    else:
        # poly: store relative polygon coords; pixel coords will be computed per-frame
        poly_rel = zone.get('coords', [])

    # Reset capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print('Starting detection. Press q to quit.')
    # Overlay state: show last detected action + object id (no timestamp)
    overlay_text = ""
    overlay_expire = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        # if poly zone, compute pixel coordinates for this frame size
        if zone.get('type') != 'rect':
            poly_px = [(int(x * W), int(y * H)) for (x, y) in poly_rel]

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (args.size, args.size), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        # collect boxes
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > args.conf and class_id == person_class_id:
                    # scale to frame size
                    cx = int(detection[0] * W)
                    cy = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.nms)
        centroids = []
        final_boxes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                cx = int(x + w/2)
                cy = int(y + h/2)
                centroids.append((cx, cy))
                # store size as (w,h) for fall detection
                final_boxes.append((x, y, w, h, confidences[i]))

        # sizes aligned with centroids for tracker (use h,w)
        sizes = []
        for (x, y, w, h, conf) in final_boxes:
            sizes.append((w, h))

        objects = tracker.update(centroids, input_sizes=sizes)

        # Map centroids -> object id for quick lookup
        id_by_centroid = {}
        for oid, centroid in objects.items():
            id_by_centroid[centroid] = oid

        # For drawing: find for each final box the matching object id by nearest centroid
        for (x, y, w, h, conf) in final_boxes:
            cx = int(x + w/2)
            cy = int(y + h/2)
            # find nearest object id
            best_id = None
            best_dist = float('inf')
            for oid, cent in objects.items():
                d = math.hypot(cent[0]-cx, cent[1]-cy)
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            # Draw bounding box and info
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"ID {best_id} {conf:.2f}" if best_id is not None else f"{conf:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # overlay centroid
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # Heuristics if we have a matched id
            if best_id is not None:
                acc = tracker.get_acceleration(best_id)
                still_time = tracker.time_still(best_id)
                speed = tracker.get_speed(best_id)
                height_ratio = tracker.get_height_ratio_change(best_id, seconds=1.0)
                vx, vy = tracker.get_velocity(best_id)

                # FALL detection: combination of rapid descent, significant height drop or horizontal aspect, then stillness
                fall_candidate = False
                if vy > FALL_DESCENT_SPEED and height_ratio < FALL_HEIGHT_RATIO:
                    fall_candidate = True
                # also if aspect ratio indicates horizontal
                # get last known size
                last_size = tracker.history.get(best_id, [])[-1][2] if tracker.history.get(best_id) else None
                if last_size is not None:
                    w_last, h_last = last_size
                    if h_last > 0 and (float(w_last) / float(h_last)) > ASPECT_RATIO_THRESHOLD:
                        fall_candidate = True

                if fall_candidate and tracker.time_still(best_id, time_window=FALL_STILLNESS_TIME, dist_threshold=STILLNESS_DISTANCE_THRESHOLD) >= FALL_STILLNESS_TIME:
                    text = f"ALERT: fall detected (ID {best_id}) height_ratio={height_ratio:.2f} vy={vy:.1f} px/s"
                    # update overlay (no timestamp)
                    overlay_text = f"FALL (ID {best_id})"
                    overlay_expire = time.time() + OVERLAY_DURATION
                    log_alert(text)
                    cv2.putText(frame, "FALL", (x, y+h+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Sudden acceleration (use CLI/configurable threshold)
                if acc > args.acc_threshold:
                    text = f"ALERT: sudden acceleration (ID {best_id}) acc={acc:.1f} px/s^2"
                    # update overlay
                    overlay_text = f"SUDDEN (ID {best_id})"
                    overlay_expire = time.time() + OVERLAY_DURATION
                    log_alert(text)
                    cv2.putText(frame, "SUDDEN", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Long stillness
                if still_time >= STILLNESS_TIME_THRESHOLD:
                    text = f"ALERT: long stillness (ID {best_id}) time={still_time:.1f}s"
                    # update overlay
                    overlay_text = f"STILLNESS (ID {best_id})"
                    overlay_expire = time.time() + OVERLAY_DURATION
                    log_alert(text)
                    cv2.putText(frame, "STILLNESS", (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Restricted zone
                in_restricted = False
                if zone.get('type') == 'rect':
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        in_restricted = True
                else:
                    # polygon check
                    if len(poly_px) >= 3:
                        contour = np.array(poly_px, dtype=np.int32)
                        in_restricted = (cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0)

                if in_restricted:
                    text = f"ALERT: restricted zone entered (ID {best_id})"
                    # update overlay
                    overlay_text = f"RESTRICTED (ID {best_id})"
                    overlay_expire = time.time() + OVERLAY_DURATION
                    log_alert(text)
                    cv2.putText(frame, "RESTRICTED", (x, y+h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Draw restricted zone (rect or polygon)
        if zone.get('type') == 'rect':
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "Restricted Zone", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        else:
            # draw polygon
            if len(poly_px) >= 3:
                pts = np.array(poly_px, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0,0,255), thickness=2)
                # place label near first vertex
                tx, ty = poly_px[0]
                cv2.putText(frame, "Restricted Zone", (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # Draw overlay box at top-left if active
        if overlay_text and time.time() < overlay_expire:
            # text rendering parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 1
            padding_x = 8
            padding_y = 6
            # compute text size
            (text_w, text_h), baseline = cv2.getTextSize(overlay_text, font, scale, thickness)
            box_w = text_w + padding_x * 2
            box_h = text_h + padding_y * 2 + baseline
            # background rectangle (filled)
            cv2.rectangle(frame, (5, 5), (5 + box_w, 5 + box_h), (50, 50, 50), -1)
            # text
            text_x = 5 + padding_x
            text_y = 5 + padding_y + text_h
            cv2.putText(frame, overlay_text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Suspicious Behavior Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
