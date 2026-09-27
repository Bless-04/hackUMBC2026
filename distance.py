"""
distance.py — GuideSense Camera-Only Distance Estimation
=========================================================
Estimates physical distance to detected objects using single-camera bounding box geometry
and the pinhole camera projection model.

Formula:
  distance_cm = (real_height_cm * FOCAL_LENGTH_PX) / bbox_height_px
  distance_m  = distance_cm / 100.0

Provides a distance estimate (metres) for the SensorFrame consumed by FusionEngine.
"""

from __future__ import annotations

from fusion import NEAR_THRESHOLD_M, Detection

# ---------------------------------------------------------------------------
# Calibrated Constants & Thresholds
# ---------------------------------------------------------------------------

# Camera focal length in pixels (calibrated at 640x480 resolution)
FOCAL_LENGTH_PX: float = 600.0

# Urgent safety threshold matching fusion.py's NEAR_THRESHOLD_M (0.60 m = 60.0 cm)
URGENT_THRESHOLD_CM: float = NEAR_THRESHOLD_M * 100.0  # 60.0 cm

# Default fallback distance (metres) when no objects are detected in frame (FAR zone)
DEFAULT_FAR_M: float = 4.0

# Default object height in cm if a label is not found in the lookup table
DEFAULT_OBJECT_HEIGHT_CM: float = 100.0


# ---------------------------------------------------------------------------
# Real-World Object Height Lookup Table (COCO labels, in centimeters)
# ---------------------------------------------------------------------------

OBJECT_HEIGHTS_CM: dict[str, float] = {
    # People & Vehicles
    "person": 170.0,
    "bicycle": 100.0,
    "motorcycle": 110.0,
    "car": 150.0,
    "bus": 300.0,
    "truck": 250.0,
    "train": 350.0,
    "boat": 150.0,
    "airplane": 300.0,

    # Animals & Indoor Pets
    "animal": 55.0,
    "dog": 60.0,
    "cat": 30.0,
    "bird": 20.0,
    "horse": 160.0,
    "sheep": 80.0,
    "cow": 140.0,

    # Indoor Furniture & Household
    "chair": 85.0,
    "couch": 85.0,
    "sofa": 85.0,
    "bed": 60.0,
    "dining table": 75.0,
    "table": 75.0,
    "toilet": 75.0,
    "screen": 60.0,
    "laptop": 25.0,
    "sink": 80.0,
    "refrigerator": 175.0,
    "oven": 85.0,
    "microwave": 35.0,
    "potted plant": 40.0,
    "bench": 80.0,

    # Street & Navigation Obstacles
    "traffic light": 100.0,
    "fire hydrant": 80.0,
    "stop sign": 75.0,
    "parking meter": 120.0,

    # Common Accessories & Objects
    "backpack": 45.0,
    "umbrella": 70.0,
    "handbag": 30.0,
    "suitcase": 60.0,
    "bottle": 25.0,
    "cup": 12.0,
    "book": 25.0,
    "clock": 30.0,
    "vase": 30.0,
    "cell phone": 15.0,
}


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def get_real_height_cm(label: str) -> float:
    """
    Look up estimated real-world height (in cm) for a given object class label.
    Case-insensitive with fallback to DEFAULT_OBJECT_HEIGHT_CM.
    """
    return OBJECT_HEIGHTS_CM.get(label.lower().strip(), DEFAULT_OBJECT_HEIGHT_CM)


def estimate_distance_cm(
    bbox_height_px: float,
    real_height_cm: float = 170.0,
    focal_length_px: float = FOCAL_LENGTH_PX,
) -> float:
    """
    Estimate distance to an object in centimeters using the pinhole camera formula:
        distance_cm = (real_height_cm * focal_length_px) / bbox_height_px

    Args:
        bbox_height_px: Height of the detection bounding box in pixels.
        real_height_cm: Expected physical height of the object in cm.
        focal_length_px: Camera focal length in pixels.

    Returns:
        Estimated distance in centimeters (returns float('inf') if bbox_height_px <= 0).
    """
    if bbox_height_px <= 0:
        return float("inf")
    return (real_height_cm * focal_length_px) / float(bbox_height_px)


def estimate_distance_m(
    bbox_height_px: float,
    real_height_cm: float = 170.0,
    focal_length_px: float = FOCAL_LENGTH_PX,
) -> float:
    """
    Estimate distance to an object in metres.

    Args:
        bbox_height_px: Height of the detection bounding box in pixels.
        real_height_cm: Expected physical height of the object in cm.
        focal_length_px: Camera focal length in pixels.

    Returns:
        Estimated distance in metres.
    """
    return estimate_distance_cm(bbox_height_px, real_height_cm, focal_length_px) / 100.0


def enrich_detections(
    detections: list[Detection],
    focal_length_px: float = FOCAL_LENGTH_PX,
    default_far_m: float = DEFAULT_FAR_M,
) -> float:
    """
    Processes a list of Detection objects, computes the estimated distance (metres)
    for each detected object from its bounding box height, and returns the MINIMUM
    distance across all detections.

    This estimate becomes SensorFrame.distance_m.

    Args:
        detections: List of Detection instances from the vision reader.
        focal_length_px: Camera focal length in pixels.
        default_far_m: Default distance returned if no detections exist or all bboxes are invalid.

    Returns:
        Minimum estimated distance in metres.
    """
    if not detections:
        return default_far_m

    estimated_distances: list[float] = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bbox_height_px = abs(y2 - y1)
        if bbox_height_px <= 0:
            continue

        real_h = get_real_height_cm(det.label)
        dist_m = estimate_distance_m(
            bbox_height_px=bbox_height_px,
            real_height_cm=real_h,
            focal_length_px=focal_length_px,
        )
        estimated_distances.append(dist_m)

    if not estimated_distances:
        return default_far_m

    return min(estimated_distances)
