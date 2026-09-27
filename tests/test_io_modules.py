"""Focused tests for the IT-owned vision, audio, haptic, and logging adapters."""

from __future__ import annotations

import csv
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.fusion import Detection, FusionAction, FusionResult, SensorFrame, Zone
from core.state_machine import SystemState


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

    monkeypatch.setattr("hardware.eleven_audio.ElevenLabsVoice", FakeEleven)
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


def test_system_audio_alert_is_idempotent_and_stops(monkeypatch):
    import haptics

    played = threading.Event()
    monkeypatch.setattr(haptics, "play_wav_bytes", lambda *_args, **_kwargs: played.set() or True)
    output = haptics.HapticOutput()
    output.buzzer_on()
    worker = output._thread
    output.buzzer_on()
    assert played.wait(1.0)
    assert output._thread is worker
    assert output.is_active
    output.buzzer_off()
    output.buzzer_off()
    output.cleanup()
    output.cleanup()

    assert not output.is_active
    assert not worker.is_alive()


def test_system_wav_playback_on_windows(monkeypatch):
    from audio_playback import play_wav_bytes

    calls = []
    monkeypatch.setattr("audio_playback.platform.system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(
        SND_MEMORY=4, PlaySound=lambda data, flags: calls.append((data, flags))))
    assert play_wav_bytes(b"RIFF-test")
    assert calls == [(b"RIFF-test", 4)]


def test_system_wav_playback_on_macos_and_linux(monkeypatch):
    import audio_playback
    from audio_playback import play_wav_bytes

    commands = []
    monkeypatch.setattr(audio_playback.subprocess, "run", lambda command, **_kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0))
    for system, player, expected in [("Darwin", "afplay", "afplay"),
                                     ("Linux", "paplay", "paplay"),
                                     ("Linux", "aplay", "aplay")]:
        monkeypatch.setattr(audio_playback.platform, "system", lambda value=system: value)
        monkeypatch.setattr(audio_playback.shutil, "which",
                            lambda name, chosen=player: f"/usr/bin/{name}" if name == chosen else None)
        assert play_wav_bytes(b"RIFF-test")
        assert expected in commands[-1][0]
        assert not Path(commands[-1][-1]).exists()


def test_unavailable_system_player_allows_speech_fallback(monkeypatch):
    import audio_playback

    monkeypatch.setattr(audio_playback.platform, "system", lambda: "Linux")
    monkeypatch.setattr(audio_playback.shutil, "which", lambda _name: None)
    assert not audio_playback.play_wav_bytes(b"RIFF-test")


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
    assert rows[1][2] == "person(0.85,LEFT)"
    assert rows[2][2] == "chair(0.85,LEFT)"
