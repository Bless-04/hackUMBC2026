"""
tests/test_distance_estimator.py — Tests for Monocular Distance Estimator
==========================================================================
Verifies that bounding box height ratios map correctly to NEAR, MID, and FAR zones.
"""

from __future__ import annotations

import pytest
from fusion import Detection
from distance_estimator import CameraDistanceEstimator


@pytest.fixture
def estimator():
    # 480px tall frame
    return CameraDistanceEstimator(frame_height=480, fallback_distance_m=3.5)


def test_empty_detections_returns_fallback(estimator):
    estimator.update_detections([])
    assert estimator.read() == 3.5


def test_tall_bounding_box_triggers_near_zone(estimator):
    """Bbox height > 65% of frame -> distance < 0.60m (NEAR / URGENT)."""
    # 400px out of 480px = ~83%
    close_det = Detection("person", 0.90, bbox=(100, 40, 300, 440))
    estimator.update_detections([close_det])
    dist = estimator.read()
    assert dist < 0.60, f"Expected near zone (<0.6m), got {dist}m"


def test_medium_bounding_box_triggers_mid_zone(estimator):
    """Bbox height ~40% of frame -> distance between 0.60m and 2.00m (MID)."""
    # 200px out of 480px = ~41%
    mid_det = Detection("person", 0.85, bbox=(150, 100, 300, 300))
    estimator.update_detections([mid_det])
    dist = estimator.read()
    assert 0.60 <= dist <= 2.00, f"Expected mid zone (0.6 - 2.0m), got {dist}m"


def test_small_bounding_box_triggers_far_zone(estimator):
    """Bbox height < 20% of frame -> distance > 2.00m (FAR / SILENT)."""
    # 60px out of 480px = 12.5%
    far_det = Detection("person", 0.80, bbox=(200, 100, 260, 160))
    estimator.update_detections([far_det])
    dist = estimator.read()
    assert dist > 2.00, f"Expected far zone (>2.0m), got {dist}m"


def test_multiple_objects_picks_closest(estimator):
    """Closest object (largest bounding box) determines the safety distance."""
    far_det = Detection("chair", 0.75, bbox=(50, 50, 100, 100))        # 50px tall
    close_det = Detection("person", 0.90, bbox=(150, 50, 350, 400))     # 350px tall (~73%)
    estimator.update_detections([far_det, close_det])
    dist = estimator.read()
    assert dist < 0.60
