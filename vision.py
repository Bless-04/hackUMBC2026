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

# Indoor navigation label mapping:
# - Broaden specific animals (dog, cat, bird, horse, sheep, cow) -> "animal"
# - Filter out outdoor vehicles (airplane, boat, motorcycle, bus, train) -> None (ignored)
# - Keep indoor relevant objects (person, chair, dining table, couch, tv, bottle, potted plant, etc.)
INDOOR_LABEL_MAP: dict[str, str | None] = {
    "airplane": None,       # Filtered out (not applicable indoors)
    "boat": None,           # Filtered out (not applicable indoors)
    "motorcycle": None,     # Filtered out (not applicable indoors)
    "bus": None,            # Filtered out (not applicable indoors)
    "train": None,          # Filtered out (not applicable indoors)
    "dog": "animal",        # Broadened to animal
    "cat": "animal",        # Broadened to animal
    "bird": "animal",       # Broadened to animal
    "horse": "animal",      # Broadened to animal
    "sheep": "animal",      # Broadened to animal
    "cow": "animal",        # Broadened to animal
}


def _model_path(value: str | Path) -> Path:
    """Resolve default model files next to this module, independent of cwd."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (Path(__file__).resolve().parent / path).resolve()


def _ensure_model_files(config_path: Path, weights_path: Path) -> None:
    """Download default MobileNet-SSD Caffe files if missing."""
    import urllib.request
    urls = {
        config_path: "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
        weights_path: "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel",
    }
    for target, url in urls.items():
        if not target.is_file():
            print(f"[VisionReader] Downloading missing model file {target.name}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as f:
                    f.write(resp.read())
                print(f"[VisionReader] Downloaded {target.name} successfully.")
            except Exception as exc:
                print(f"[VisionReader] Warning: Auto-download of {target.name} failed ({exc}).")


class VisionReader:
    """Capture webcam frames and turn vision output into detections."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        confidence_floor: float = CONFIDENCE_FLOOR,
        *,
        model_config: str | Path = MODEL_CONFIG,
        model_weights: str | Path = MODEL_WEIGHTS,
        max_detections: int = MAX_DETECTIONS,
        show_preview: bool = False,
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
        if missing and (str(model_config) == MODEL_CONFIG or str(model_weights) == MODEL_WEIGHTS):
            _ensure_model_files(config_path, weights_path)
            missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MobileNet-SSD model file(s) not found: " + ", ".join(missing)
            )
        self._cv2: Any = cv2
        self._confidence_floor = float(confidence_floor)
        self._max_detections = max_detections
        self._show_preview = show_preview
        self._closed = False
        self._warned_capture_failure = False
        self._latest_frame: Any | None = None

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

    @property
    def latest_frame(self) -> Any | None:
        """Returns the most recent raw BGR camera frame."""
        return self._latest_frame

    def read(self) -> list[Detection]:
        """Return detections for one frame, or ``[]`` if capture temporarily fails."""
        if self._closed or self._cap is None or not self._cap.isOpened():
            return []

        try:
            ok, frame = self._cap.read()
        except Exception as exc:  # camera drivers can fail transiently
            self._warn_capture_failure(str(exc))
            return []

        if not ok or frame is None:
            self._warn_capture_failure("no frame returned")
            return []

        self._latest_frame = frame
        self._warned_capture_failure = False
        detections = self._detect(frame)
        if self._show_preview:
            self._render_preview(frame, detections)
        return detections

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

            raw_label = MODEL_CLASSES[class_index]
            # Map label for indoor assistive demo; skip if filtered out
            final_label = INDOOR_LABEL_MAP.get(raw_label, raw_label)
            if final_label is None:
                continue

            results.append(
                Detection(
                    label=final_label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    timestamp=captured_at,
                    frame_width=width,
                )
            )

        results.sort(key=lambda detection: detection.confidence, reverse=True)
        return results[: self._max_detections]

    def _render_preview(self, frame: Any, detections: list[Detection]) -> None:
        """Display a live camera preview with bounding boxes and labels."""
        cv2 = self._cv2
        preview = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dir_str = f" [{detection.direction.name}]" if detection.direction else ""
            label = f"{detection.label}{dir_str} ({detection.confidence:.2f})"
            cv2.putText(
                preview,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("GuideSense - Vision Preview", preview)
        cv2.waitKey(1)

    def _warn_capture_failure(self, detail: str) -> None:
        if not self._warned_capture_failure:
            print(f"[VisionReader] camera frame unavailable: {detail}")
            self._warned_capture_failure = True

    def close(self) -> None:
        """Release the camera and preview window. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_cap") and self._cap is not None and hasattr(self._cap, "release"):
            self._cap.release()
        if self._show_preview and hasattr(self, "_cv2"):
            self._cv2.destroyAllWindows()

    def __enter__(self) -> "VisionReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GuideSense vision benchmark")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="webcam index")
    parser.add_argument("--gui", action="store_true", help="show the camera preview")
    args = parser.parse_args()

    reader = VisionReader(camera_index=args.camera, show_preview=args.gui)
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
