"""
vision.py — GuideSense Camera Capture & Object Detection Module
================================================================
Captures live frames from the webcam and detects objects (person, chair, etc.)
using OpenCV DNN (MobileNet-SSD), OpenCV HOG Person Detector, or Haar Cascades.

Data contract with senior:
  vision.VisionReader().read() -> list[Detection]

  Detection fields:
    detection.label       str    COCO class name, lowercase (e.g. "person", "chair")
    detection.confidence  float  0.0–1.0
    detection.bbox        tuple  (x1, y1, x2, y2) pixels in raw frame
    detection.timestamp   float  time.monotonic()

How to test standalone:
  python3 vision.py [--camera 0] [--gui]
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from fusion import Detection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX     = 0          # Default webcam index
INPUT_WIDTH      = 300        # MobileNet-SSD input width
INPUT_HEIGHT     = 300        # MobileNet-SSD input height
CONFIDENCE_FLOOR = 0.40       # Pre-filter detections before fusion gate
MAX_DETECTIONS   = 5          # Cap detections per frame for performance

COCO_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


# ---------------------------------------------------------------------------
# VisionReader
# ---------------------------------------------------------------------------

class VisionReader:
    """
    Captures live webcam frames and returns detected objects as Detection dataclass objects.
    Seamlessly falls back between MobileNet-SSD (Caffe) and OpenCV's built-in HOG detector.
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        confidence_floor: float = CONFIDENCE_FLOOR,
        show_preview: bool = False,
    ) -> None:
        self.camera_index = camera_index
        self.confidence_floor = confidence_floor
        self.show_preview = show_preview
        self._cap = None
        self._net = None
        self._hog = None
        self._face_cascade = None
        self._latest_frame = None

        self._init_camera()
        self._init_detectors()

    def _init_camera(self) -> None:
        try:
            import cv2
            self._cv2 = cv2
            self._cap = cv2.VideoCapture(self.camera_index)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[VisionReader] Webcam connected (index={self.camera_index}, 640x480)")
            else:
                print(f"[VisionReader] WARNING: Could not open camera at index {self.camera_index}")
        except ImportError:
            print("[VisionReader] ERROR: OpenCV (cv2) is not installed")

    def _init_detectors(self) -> None:
        if not hasattr(self, "_cv2"):
            return

        cv2 = self._cv2

        # 1. Try MobileNet-SSD Caffe model files
        proto = "MobileNetSSD_deploy.prototxt"
        caffe = "MobileNetSSD_deploy.caffemodel"
        if os.path.exists(proto) and os.path.exists(caffe):
            try:
                self._net = cv2.dnn.readNetFromCaffe(proto, caffe)
                print("[VisionReader] Loaded MobileNet-SSD Caffe detector")
                return
            except Exception as e:
                print(f"[VisionReader] Could not load Caffe model: {e}")

        # 2. Built-in OpenCV HOG Person Detector (No download required)
        try:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[VisionReader] Loaded OpenCV Built-In HOG Person Detector")
        except Exception as e:
            print(f"[VisionReader] HOG detector init error: {e}")

        # 3. Built-in Haar Cascade Face Detector as supplementary anchor
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

    def read(self) -> list[Detection]:
        """
        Grabs the latest camera frame and returns Detection objects.
        Non-blocking, returns [] on frame drop.
        """
        if self._cap is None or not self._cap.isOpened():
            return []

        cv2 = self._cv2
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return []

        self._latest_frame = frame
        now = time.monotonic()
        detections: list[Detection] = []
        h, w = frame.shape[:2]

        # Path A: MobileNet-SSD
        if self._net is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)),
                    0.007843, (INPUT_WIDTH, INPUT_HEIGHT), 127.5,
                )
                self._net.setInput(blob)
                detections_raw = self._net.forward()

                for i in range(detections_raw.shape[2]):
                    conf = float(detections_raw[0, 0, i, 2])
                    if conf < self.confidence_floor:
                        continue
                    class_idx = int(detections_raw[0, 0, i, 1])
                    if 0 <= class_idx < len(COCO_CLASSES):
                        label = COCO_CLASSES[class_idx].lower()
                        if label == "background":
                            continue
                        box = detections_raw[0, 0, i, 3:7] * [w, h, w, h]
                        x1, y1, x2, y2 = box.astype(int)
                        # Clamp to frame bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if (y2 - y1) > 10 and (x2 - x1) > 10:
                            detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2), timestamp=now))
            except Exception as e:
                print(f"[VisionReader] DNN Detection error: {e}")

        # Path B: OpenCV HOG Person Detector
        elif self._hog is not None:
            try:
                # Resize slightly for faster inference
                boxes, weights = self._hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05,
                )
                for (x, y, bw, bh), conf in zip(boxes, weights):
                    conf_float = float(conf)
                    # Normalize HOG SVM distance margin (typically 0.0 to 2.5) to 0.5 - 0.95
                    norm_conf = min(0.95, max(0.50, 0.50 + conf_float * 0.2))
                    if norm_conf >= self.confidence_floor:
                        detections.append(Detection(
                            label="person",
                            confidence=round(norm_conf, 2),
                            bbox=(int(x), int(y), int(x + bw), int(y + bh)),
                            timestamp=now,
                        ))
            except Exception as e:
                print(f"[VisionReader] HOG inference error: {e}")

        # Path C: Fallback to Face Cascade (extrapolate to person bounding box)
        if not detections and self._face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                for (fx, fy, fw, fh) in faces:
                    # Extrapolate full person height (~6.5x face height)
                    px1 = max(0, fx - int(fw * 0.5))
                    py1 = max(0, fy - int(fh * 0.2))
                    px2 = min(w, fx + int(fw * 1.5))
                    py2 = min(h, fy + int(fh * 5.0))
                    detections.append(Detection(
                        label="person",
                        confidence=0.75,
                        bbox=(px1, py1, px2, py2),
                        timestamp=now,
                    ))
            except Exception:
                pass

        if self.show_preview and self._latest_frame is not None:
            self._render_preview(detections)

        return detections[:MAX_DETECTIONS]

    def _render_preview(self, detections: list[Detection]) -> None:
        """Renders live camera view with bounding box overlays and labels."""
        cv2 = self._cv2
        disp = self._latest_frame.copy()

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{d.label} ({d.confidence:.2f})"
            cv2.putText(disp, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("GuideSense — Vision Preview", disp)
        cv2.waitKey(1)

    def close(self) -> None:
        """Release camera and windows."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        if hasattr(self, "_cv2"):
            self._cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GuideSense Vision Module Standalone Test")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--gui", action="store_true", help="Display visual OpenCV preview window")
    args = parser.parse_args()

    reader = VisionReader(camera_index=args.camera, show_preview=args.gui)
    print("Running GuideSense Vision detector. Press Ctrl-C to stop.\n")

    try:
        while True:
            t0 = time.monotonic()
            dets = reader.read()
            dt = time.monotonic() - t0
            fps = 1.0 / dt if dt > 0 else 0

            if dets:
                for d in dets:
                    bbox_h = abs(d.bbox[3] - d.bbox[1])
                    print(f"[{time.strftime('%H:%M:%S')}] {d.label:<12} conf={d.confidence:.2f}  bbox={d.bbox}  height={bbox_h}px  ({fps:.1f} FPS)")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] (scanning...)  ({fps:.1f} FPS)", end="\r", flush=True)

            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.close()
        print("\n[VisionReader] Stopped cleanly.")
