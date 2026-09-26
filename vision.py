"""
Camera capture and MobileNet-SSD object detection.

``VisionReader`` owns the webcam and OpenCV DNN model. Its public contract is
small on purpose: call :meth:`read` once per sensing tick and :meth:`close` at
shutdown. A failed frame read is treated as a temporary sensor outage and
returns an empty list rather than taking down the safety loop.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from fusion import Detection

CAMERA_INDEX = 0
INPUT_WIDTH = 300
INPUT_HEIGHT = 300
CONFIDENCE_FLOOR = 0.40
MAX_DETECTIONS = 5
MODEL_CONFIG = "MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "MobileNetSSD_deploy.caffemodel"

# The recommended Caffe MobileNet-SSD checkpoint was trained on the 20 PASCAL
# VOC classes. The canonicalized spellings below match fusion.OBJECT_PRIORITY
# (notably ``motorcycle`` and ``couch``).
MODEL_CLASSES = [
    "background",
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "couch",
    "train",
    "tv",
]
# Backward-compatible name retained for teammates that imported the original
# scaffold constant.
COCO_CLASSES = MODEL_CLASSES


def _model_path(value: str | Path) -> Path:
    """Resolve default model files next to this module, independent of cwd."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


class VisionReader:
    """Capture webcam frames and turn MobileNet-SSD output into detections."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        confidence_floor: float = CONFIDENCE_FLOOR,
        *,
        model_config: str | Path = MODEL_CONFIG,
        model_weights: str | Path = MODEL_WEIGHTS,
        max_detections: int = MAX_DETECTIONS,
    ) -> None:
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be between 0.0 and 1.0")
        if max_detections < 1:
            raise ValueError("max_detections must be at least 1")

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on host setup
            raise RuntimeError(
                "OpenCV is required for VisionReader; install opencv-python"
            ) from exc

        config_path = _model_path(model_config)
        weights_path = _model_path(model_weights)
        missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MobileNet-SSD model file(s) not found: " + ", ".join(missing)
            )

        self._cv2: Any = cv2
        self._confidence_floor = float(confidence_floor)
        self._max_detections = max_detections
        self._closed = False
        self._warned_capture_failure = False

        try:
            self._net = cv2.dnn.readNetFromCaffe(str(config_path), str(weights_path))
        except Exception as exc:
            raise RuntimeError(f"Could not load MobileNet-SSD model: {exc}") from exc

        self._cap = cv2.VideoCapture(camera_index)
        if hasattr(self._cap, "isOpened") and not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Could not open camera index {camera_index}")

        print(
            f"[VisionReader] camera={camera_index} model={weights_path.name} "
            f"confidence>={self._confidence_floor:.2f}"
        )

    def read(self) -> list[Detection]:
        """Return detections for one frame, or ``[]`` if capture temporarily fails."""
        if self._closed:
            return []

        try:
            ok, frame = self._cap.read()
        except Exception as exc:  # camera drivers can fail transiently
            self._warn_capture_failure(str(exc))
            return []

        if not ok or frame is None:
            self._warn_capture_failure("no frame returned")
            return []

        self._warned_capture_failure = False
        return self._detect(frame)

    def _detect(self, frame: Any) -> list[Detection]:
        """Run inference for a captured frame (kept separate for focused tests)."""
        cv2 = self._cv2
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        blob = cv2.dnn.blobFromImage(
            resized,
            0.007843,
            (INPUT_WIDTH, INPUT_HEIGHT),
            127.5,
        )
        self._net.setInput(blob)
        raw = self._net.forward()
        captured_at = time.monotonic()
        results: list[Detection] = []

        if raw is None or len(raw.shape) < 3:
            return results

        for index in range(raw.shape[2]):
            confidence = float(raw[0, 0, index, 2])
            if not math.isfinite(confidence) or confidence < self._confidence_floor:
                continue

            class_index = int(raw[0, 0, index, 1])
            if class_index <= 0 or class_index >= len(MODEL_CLASSES):
                continue

            scaled = raw[0, 0, index, 3:7] * [width, height, width, height]
            x1, y1, x2, y2 = (int(value) for value in scaled)
            x1 = min(max(x1, 0), width - 1)
            y1 = min(max(y1, 0), height - 1)
            x2 = min(max(x2, 0), width - 1)
            y2 = min(max(y2, 0), height - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                Detection(
                    label=MODEL_CLASSES[class_index],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    timestamp=captured_at,
                )
            )

        # The DNN output order is not a documented ranking. Keeping the most
        # confident results makes MAX_DETECTIONS deterministic and useful.
        results.sort(key=lambda detection: detection.confidence, reverse=True)
        return results[: self._max_detections]

    def _warn_capture_failure(self, detail: str) -> None:
        if not self._warned_capture_failure:
            print(f"[VisionReader] camera frame unavailable: {detail}")
            self._warned_capture_failure = True

    def close(self) -> None:
        """Release the camera. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._cap.release()

    def __enter__(self) -> "VisionReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    reader = VisionReader()
    print("Running detection benchmark. Ctrl-C to stop.")
    frame_count = 0
    benchmark_start = time.monotonic()
    try:
        while True:
            detections = reader.read()
            frame_count += 1
            for detection in detections:
                print(
                    f"  {detection.label:<15} conf={detection.confidence:.2f} "
                    f"bbox={detection.bbox}"
                )

            elapsed = time.monotonic() - benchmark_start
            if elapsed >= 1.0:
                print(f"  FPS={frame_count / elapsed:.1f} (target >= 10)")
                frame_count = 0
                benchmark_start = time.monotonic()
    except KeyboardInterrupt:
        print("Done.")
    finally:
        reader.close()
