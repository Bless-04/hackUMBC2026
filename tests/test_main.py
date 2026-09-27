"""Checks that the command line live path consumes camera data and fails clearly."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import main
from core.fusion import Detection


def test_live_cli_uses_camera_frame_size_for_distance(monkeypatch):
    detection = Detection("person", .93, (20, 40, 120, 200), frame_width=320)
    reader = SimpleNamespace(
        latest_frame=np.zeros((240, 320, 3), dtype=np.uint8),
        closed=False,
    )
    reader.read = lambda: [detection]
    reader.close = lambda: setattr(reader, "closed", True)
    cameras = []

    def open_camera(*, camera_index):
        cameras.append(camera_index)
        return reader

    monkeypatch.setattr("vision.vision.VisionReader", open_camera)
    estimators = []

    class Estimator:
        frame_height = 480

        def __init__(self):
            self.detections = []
            estimators.append(self)

        def update_detections(self, detections):
            self.detections = detections

        def read(self):
            return 1.5

    monkeypatch.setattr("vision.distance_estimator.CameraDistanceEstimator", Estimator)
    main.run(use_camera=True, camera_index=1, enable_logging=False,
             duration_sec=.11, verbose=False)

    assert cameras == [1]
    assert reader.closed
    assert estimators[0].frame_height == 240
    assert estimators[0].detections == [detection]


def test_live_cli_does_not_show_simulated_data_when_camera_fails(monkeypatch):
    def unavailable(*, camera_index):
        raise RuntimeError(f"Camera {camera_index} is unavailable")

    monkeypatch.setattr("vision.vision.VisionReader", unavailable)
    with pytest.raises(RuntimeError, match="Camera 1 is unavailable"):
        main.run(use_camera=True, camera_index=1, enable_logging=False, verbose=False)
