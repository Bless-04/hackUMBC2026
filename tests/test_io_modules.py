"""Focused tests for the IT-owned vision, audio, haptic, and logging adapters."""

from __future__ import annotations

import csv
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from fusion import Detection, FusionAction, FusionResult, SensorFrame, Zone
from state_machine import SystemState


def test_vision_converts_filters_clamps_and_ranks(monkeypatch, tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    raw = np.array(
        [
            [
                [
                    [0, 15, 0.91, -0.1, 0.1, 1.1, 0.9],  # person, clamp x
                    [0, 9, 0.30, 0.1, 0.1, 0.5, 0.5],  # below floor
                    [0, 18, 0.80, 0.2, 0.2, 0.8, 0.8],  # couch alias
                    [0, 99, 0.99, 0.1, 0.1, 0.5, 0.5],  # invalid class
                ]
            ]
        ],
        dtype=float,
    )

    class FakeNet:
        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            return raw

    class FakeCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    fake_dnn = SimpleNamespace(
        readNetFromCaffe=lambda *_: FakeNet(),
        blobFromImage=lambda *args: args[0],
    )
    fake_cv2 = SimpleNamespace(
        dnn=fake_dnn,
        VideoCapture=lambda _: capture,
        resize=lambda image, _size: image,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    config = tmp_path / "model.prototxt"
    weights = tmp_path / "model.caffemodel"
    config.touch()
    weights.touch()

    from vision import VisionReader

    reader = VisionReader(
        confidence_floor=0.4,
        model_config=config,
        model_weights=weights,
    )
    detections = reader.read()
    reader.close()
    reader.close()

    assert [detection.label for detection in detections] == ["person", "couch"]
    assert detections[0].bbox == (0, 10, 199, 90)
    assert capture.released


def test_audio_queues_speech_on_one_worker(monkeypatch):
    spoken: list[str] = []
    finished = threading.Event()

    class FakeEleven:
        is_available = False

    class FakeEngine:
        def setProperty(self, _name, _value):
            pass

        def say(self, text):
            spoken.append(text)

        def runAndWait(self):
            finished.set()

    monkeypatch.setitem(
        sys.modules,
        "eleven_audio",
        SimpleNamespace(ElevenLabsVoice=lambda: FakeEleven()),
    )
    monkeypatch.setitem(sys.modules, "pyttsx3", SimpleNamespace(init=FakeEngine))

    from audio import AudioOutput

    output = AudioOutput()
    started = time.monotonic()
    output.speak("person")
    assert time.monotonic() - started < 0.05
    assert finished.wait(1.0)
    output.close()
    output.close()

    assert spoken == ["person"]


def test_serial_haptics_are_idempotent_and_do_not_close_shared_connection():
    class FakeSerial:
        def __init__(self):
            self.writes: list[bytes] = []
            self.closed = False

        def write(self, payload):
            self.writes.append(payload)

        def close(self):
            self.closed = True

    from haptics import HapticOutput

    serial = FakeSerial()
    output = HapticOutput(mode="serial", serial_connection=serial)
    output.buzzer_on()
    output.buzzer_on()
    output.buzzer_off()
    output.buzzer_off()
    output.cleanup()

    assert serial.writes == [b"BUZZ_ON\n", b"BUZZ_OFF\n"]
    assert not serial.closed


def test_logger_appends_without_repeating_header(tmp_path):
    from logger import CSV_HEADERS, EventLogger

    path = tmp_path / "events.csv"
    for label in ("person", "chair"):
        logger = EventLogger(path, flush_every_n=1)
        frame = SensorFrame(
            distance_m=1.5,
            detections=[Detection(label, 0.85, (0, 0, 1, 1))],
        )
        result = FusionResult(
            action=FusionAction.INFORMATIVE,
            label=label,
            distance_m=1.5,
            zone=Zone.MID,
            reason="test",
        )
        logger.log_event(frame, result, SystemState.INFORMATIVE)
        logger.close()
        logger.close()

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == CSV_HEADERS
    assert len(rows) == 3
    assert rows[1][2] == "person(0.85)"
    assert rows[2][2] == "chair(0.85)"
