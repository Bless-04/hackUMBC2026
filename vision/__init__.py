"""
vision — GuideSense Computer Vision & Distance Estimation
==========================================================
Webcam capture, MobileNet-SSD local inference, and monocular distance estimation.
"""

from vision.distance import (
    DEFAULT_FAR_M,
    DEFAULT_OBJECT_HEIGHT_CM,
    FOCAL_LENGTH_PX,
    OBJECT_HEIGHTS_CM,
    URGENT_THRESHOLD_CM,
    estimate_distance_cm,
    estimate_distance_m,
    get_real_height_cm,
)
from vision.distance_estimator import CameraDistanceEstimator
from vision.vision import (
    COCO_CLASSES,
    CONFIDENCE_FLOOR,
    INDOOR_LABEL_MAP,
    MAX_DETECTIONS,
    MODEL_CLASSES,
    VisionReader,
)

__all__ = [
    "COCO_CLASSES",
    "CONFIDENCE_FLOOR",
    "CameraDistanceEstimator",
    "DEFAULT_FAR_M",
    "DEFAULT_OBJECT_HEIGHT_CM",
    "FOCAL_LENGTH_PX",
    "INDOOR_LABEL_MAP",
    "MAX_DETECTIONS",
    "MODEL_CLASSES",
    "OBJECT_HEIGHTS_CM",
    "URGENT_THRESHOLD_CM",
    "VisionReader",
    "estimate_distance_cm",
    "estimate_distance_m",
    "get_real_height_cm",
]
