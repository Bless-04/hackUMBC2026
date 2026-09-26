"""
vision.py — IT Freshman's Module
==================================
Camera capture + MobileNet-SSD object detection.

OWNED BY: Freshman #2 (Information Technology)
HANDED OFF TO: Senior (CS) via the read() interface below.

Data contract with senior:
  vision.VisionReader().read() -> list[Detection]

  Detection is imported from fusion.py — do NOT redefine it here.
  Fields the senior reads:
    detection.label       str    COCO class name, lowercase  e.g. "person", "chair"
    detection.confidence  float  0.0–1.0
    detection.bbox        tuple  (x1, y1, x2, y2) pixels in the raw camera frame
    detection.timestamp   float  time.monotonic() at capture time

Model recommendation (from manual Part 8):
  MobileNet-SSD pretrained on COCO
  Framework: OpenCV DNN  (cv2.dnn.readNetFromCaffe)
  Files needed:
    MobileNetSSD_deploy.prototxt
    MobileNetSSD_deploy.caffemodel

Benchmark procedure (manual Part 8):
  1. Run benchmark.py (in this repo) to measure FPS before plugging into main loop
  2. Target >= 10 FPS at 300x300 input resolution on Pi hardware
  3. If below target: lower confidence_min in fusion.py or drop to 5 Hz tick rate

How to test standalone (before handoff):
  python -X utf8 vision.py
  → opens camera, prints detections to stdout, Ctrl-C to stop
"""

from __future__ import annotations

import time

from fusion import Detection

# ---------------------------------------------------------------------------
# Configuration — IT freshman adjusts these to match their setup
# ---------------------------------------------------------------------------

CAMERA_INDEX      = 0          # Pi camera index (usually 0)
INPUT_WIDTH       = 300        # MobileNet-SSD native input size
INPUT_HEIGHT      = 300
CONFIDENCE_FLOOR  = 0.40       # pre-filter before sending to fusion (fusion has its own gate at 0.50)
MAX_DETECTIONS    = 5          # cap detections per frame for performance

# COCO classes MobileNet-SSD was trained on (index → label)
# IT freshman: keep this list — fusion.py priority table uses these exact strings
COCO_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


# ---------------------------------------------------------------------------
# Real implementation — IT freshman fills this in
# ---------------------------------------------------------------------------

class VisionReader:
    """
    Captures one camera frame and returns a list of Detections.

    Usage:
        reader = VisionReader()
        detections = reader.read()   # call this every tick

    The senior's main.py calls reader.read() at TICK_HZ (10 Hz).
    Each call should return immediately with the latest available frame
    (don't block waiting for a new frame if the camera is slow).
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        confidence_floor: float = CONFIDENCE_FLOOR,
    ) -> None:
        # IT freshman: uncomment when model files and camera are ready
        # import cv2
        # self._cap = cv2.VideoCapture(camera_index)
        # self._net = cv2.dnn.readNetFromCaffe(
        #     "MobileNetSSD_deploy.prototxt",
        #     "MobileNetSSD_deploy.caffemodel",
        # )
        self._confidence_floor = confidence_floor
        print(f"[VisionReader] stub — camera={camera_index}  (not connected)")

    def read(self) -> list[Detection]:
        """
        Return a list of Detection objects from the latest camera frame.
        Returns [] if camera unavailable or no objects detected above threshold.

        IT freshman TODO:
            ret, frame = self._cap.read()
            if not ret:
                return []

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)),
                0.007843, (INPUT_WIDTH, INPUT_HEIGHT), 127.5,
            )
            self._net.setInput(blob)
            detections_raw = self._net.forward()

            results = []
            h, w = frame.shape[:2]
            now = time.monotonic()
            for i in range(detections_raw.shape[2]):
                conf = float(detections_raw[0, 0, i, 2])
                if conf < self._confidence_floor:
                    continue
                class_idx = int(detections_raw[0, 0, i, 1])
                label = COCO_CLASSES[class_idx].lower()
                box = detections_raw[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                results.append(Detection(label=label, confidence=conf, bbox=(x1,y1,x2,y2), timestamp=now))

            return results[:MAX_DETECTIONS]
        """
        # Stub: return empty list until camera + model are wired
        return []

    def close(self) -> None:
        """Release camera on shutdown."""
        # self._cap.release()
        pass


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reader = VisionReader()
    print("Running detection. Ctrl-C to stop.")
    try:
        while True:
            dets = reader.read()
            if dets:
                for d in dets:
                    print(f"  {d.label:<15} conf={d.confidence:.2f}  bbox={d.bbox}")
            else:
                print("  (no detections)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.close()
        print("Done.")
