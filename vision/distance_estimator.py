"""
distance_estimator.py — Monocular Distance Estimation from Camera Bounding Boxes
================================================================================
Estimates object distance in metres using the apparent vertical height of
the detected bounding box relative to the camera frame height.

Principle:
  distance ~ (focal_length * real_object_height) / bbox_pixel_height

Tuned for standard 480p/720p webcams (e.g. Logitech C270 / C920):
  - Person close (>60% frame height)   -> NEAR zone (<0.6m)  -> URGENT
  - Person mid (20% - 60% frame height) -> MID zone (0.6m - 2.0m) -> INFORMATIVE
  - Person far (<20% frame height)     -> FAR zone (>2.0m)   -> SILENT
"""

from __future__ import annotations

from typing import List

from core.fusion import Detection

# Approximate typical physical heights of objects in metres
TYPICAL_HEIGHT_M: dict[str, float] = {
    "person": 1.70,
    "animal": 0.55,
    "dog": 0.60,
    "cat": 0.30,
    "chair": 0.85,
    "dining table": 0.75,
    "couch": 0.85,
    "tv": 0.60,
    "potted plant": 0.45,
    "bottle": 0.25,
}
DEFAULT_OBJECT_HEIGHT_M = 1.00

# Default camera vertical resolution
DEFAULT_FRAME_HEIGHT = 480


class CameraDistanceEstimator:
    """
    Computes an estimated distance in metres from camera detections.
    Exposes `.read()` for each sensing tick.
    """

    def __init__(
        self,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        fallback_distance_m: float = 3.5,
    ):
        self.frame_height = frame_height
        self.fallback_distance_m = fallback_distance_m
        self._latest_detections: List[Detection] = []

    def update_detections(self, detections: List[Detection]) -> None:
        """Call this each tick with the latest detections from vision.py."""
        self._latest_detections = detections

    def estimate_distance_for_detection(self, det: Detection) -> float:
        """
        Estimate distance to a single detection based on its bounding box height.
        bbox is expected in format (x1, y1, x2, y2).
        """
        y1, y2 = det.bbox[1], det.bbox[3]
        bbox_height = max(1, abs(y2 - y1))
        height_ratio = bbox_height / float(self.frame_height)

        # Height ratio thresholds:
        # > 0.65 frame height -> very close (< 0.55m, near/urgent)
        # 0.20 - 0.65 frame height -> mid range (0.6m - 2.0m)
        # < 0.20 frame height -> far (> 2.0m)
        if height_ratio >= 0.65:
            # Scale smoothly within NEAR zone (0.3m to 0.59m)
            dist = max(0.30, 0.60 - (height_ratio - 0.65) * 0.8)
            return round(dist, 2)
        elif height_ratio >= 0.20:
            # Linear interpolation across MID zone (0.60m to 2.00m)
            norm = (0.65 - height_ratio) / (0.65 - 0.20)
            dist = 0.60 + norm * (2.00 - 0.60)
            return round(dist, 2)
        else:
            # FAR zone
            norm = max(0.0, (0.20 - height_ratio) / 0.20)
            dist = 2.00 + norm * 1.50
            return round(dist, 2)

    def read(self) -> float:
        """
        Returns estimated distance in metres to the closest detected object.
        If no objects are detected, returns fallback_distance_m (safe/far).
        """
        if not self._latest_detections:
            return self.fallback_distance_m

        # Closest object is the one with the tallest relative bounding box
        distances = [
            self.estimate_distance_for_detection(d)
            for d in self._latest_detections
            if d.confidence >= 0.40
        ]
        if not distances:
            return self.fallback_distance_m

        return min(distances)
