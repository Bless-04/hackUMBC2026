"""Dashboard tests: no physical camera, speaker, or external cloud requests."""

import csv
import io
import json
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ui import runtime
from ui.runtime import DashboardHardware, SessionBusy, SessionController, validate_config
from ui.server import make_handler


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Dashboard did not reach the expected state")


@pytest.mark.parametrize("settings", [
    None, [], {"mode": "invalid"}, {"camera": -1}, {"camera": 11},
    {"camera": True}, {"camera": 0.5}, {"voice": "true"}, {"logging": 1},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        validate_config(settings)


def test_demo_forces_hardware_and_cloud_off():
    config = validate_config(dict(mode="demo", voice=True, gemini=True,
                                  backboard=True, logging=True))
    assert not any(config[key] for key in ("voice", "gemini", "backboard"))
    assert config["logging"] is True
    assert validate_config({"mode": "camera", "gemini": True})["gemini"] is True


def test_demo_covers_far_mid_and_near():
    assert SessionController.demo_frame(0) == (3.5, [])
    assert .6 < SessionController.demo_frame(7)[0] < 2
    distance, detections = SessionController.demo_frame(9)
    assert distance < .6
    assert detections[0].label == "person"
    assert detections[0].direction is not None


@pytest.fixture
def controller():
    instance = SessionController()
    yield instance
    instance.shutdown()


def test_demo_lifecycle_and_no_external_resources(controller, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("Demo must remain local and silent"))
    for module, constructor in [("vision", "VisionReader"), ("audio", "AudioOutput"),
                                ("haptics", "HapticOutput"),
                                ("gemini_narrator", "GeminiNarrator"),
                                ("backboard_memory", "BackboardMemory")]:
        # state_machine already imports format_voice_message; these constructors stay unused.
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{constructor: forbidden}))
    controller.start({"mode": "demo", "voice": True, "gemini": True, "backboard": True})
    wait_until(lambda: controller.snapshot()["ticks"] > 0)
    assert controller.snapshot()["status"] == "running"
    assert controller.snapshot()["services"]["camera"] == "Simulated"
    with pytest.raises(SessionBusy):
        controller.start({})
    controller.shutdown()
    assert controller.snapshot()["status"] == "stopped"
    assert not controller.thread.is_alive()
    controller.start({})
    wait_until(lambda: controller.snapshot()["ticks"] > 0)
    assert controller.generation == 2
    forbidden.assert_not_called()


def test_demo_logging_uses_existing_csv_logger(controller, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "SESSION_DIR", tmp_path)
    controller.start({"logging": True})
    wait_until(lambda: controller.snapshot()["ticks"] > 1)
    controller.shutdown()
    files = list(tmp_path.glob("session-*.csv"))
    assert len(files) == 1
    with files[0].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 2
    assert rows[0]["system_state"] == "SILENT"


def test_camera_failure_is_visible_and_releases_reader(controller, monkeypatch):
    reader = Mock()
    reader.read.side_effect = RuntimeError("synthetic camera failure")
    monkeypatch.setitem(sys.modules, "vision", SimpleNamespace(VisionReader=Mock(return_value=reader)))
    controller.start({"mode": "camera"})
    wait_until(lambda: controller.snapshot()["status"] == "error")
    assert controller.snapshot()["error"]
    assert not controller.snapshot()["has_frame"]
    assert "synthetic camera failure" not in controller.snapshot()["error"]
    reader.close.assert_called_once()


def test_partial_audio_setup_is_cleaned_up(controller, monkeypatch):
    reader, audio = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "vision", SimpleNamespace(VisionReader=Mock(return_value=reader)))
    monkeypatch.setitem(sys.modules, "audio", SimpleNamespace(AudioOutput=Mock(return_value=audio)))
    monkeypatch.setitem(sys.modules, "haptics", SimpleNamespace(
        HapticOutput=Mock(side_effect=RuntimeError("no alert output"))))
    controller.start({"mode": "camera", "voice": True})
    wait_until(lambda: controller.snapshot()["status"] == "error")
    audio.close.assert_called_once()
    reader.close.assert_called_once()


def test_stale_camera_frame_is_not_shown_as_live(controller, monkeypatch):
    import numpy as np

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    reader = Mock(latest_frame=image)
    reader.read.return_value = []  # Repeated old frame simulates capture failure.
    monkeypatch.setitem(sys.modules, "vision", SimpleNamespace(VisionReader=Mock(return_value=reader)))
    controller.start({"mode": "camera"})
    wait_until(lambda: controller.snapshot()["ticks"] > 2)
    state = controller.snapshot()
    assert state["frame_width"] == 320
    assert state["frame_height"] == 240
    assert state["distance"] is None
    assert state["has_frame"] is False


def test_fresh_camera_frames_are_encoded(controller, monkeypatch):
    import numpy as np

    reader = Mock(latest_frame=None)

    def read():
        reader.latest_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return []

    reader.read.side_effect = read
    monkeypatch.setitem(sys.modules, "vision", SimpleNamespace(VisionReader=Mock(return_value=reader)))
    controller.start({"mode": "camera"})
    wait_until(lambda: controller.snapshot()["has_frame"])
    assert controller.jpeg.startswith(b"\xff\xd8")
    assert controller.snapshot()["frame_height"] == 240
    controller.shutdown()
    assert controller.jpeg is None
    reader.close.assert_called_once()


def test_late_cloud_callbacks_cannot_speak_after_stop_or_restart(controller):
    audio = Mock()
    hardware = DashboardHardware(controller, controller.generation, audio)
    hardware.announce("An object ahead.", "Gemini")
    audio.speak.assert_called_once()
    controller.stop_event.set()
    hardware.announce("After stopping", "Gemini")
    controller.stop_event.clear()
    controller.generation += 1
    hardware.announce("From the previous session", "Gemini")
    audio.speak.assert_called_once()


def test_events_are_bounded_and_export_escapes_formulas(controller):
    for number in range(205):
        controller.event("State", str(number))
    controller.event("Guidance", "=1+1")
    assert len(controller.snapshot()["events"]) == 200
    rows = list(csv.reader(io.StringIO(controller.export().decode("utf-8-sig"))))
    assert rows[0] == ["time", "category", "message"]
    assert rows[-1][-1] == "'=1+1"


@pytest.fixture
def http_server(controller):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(controller, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, path, method="GET", body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


def test_http_assets_and_private_file_isolation(http_server):
    status, body, headers = request(http_server, "/")
    assert status == 200
    assert b"test-token" in body and b"__SESSION_TOKEN__" not in body
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    for path in ("/app.css", "/app.js", "/scene.svg", "/mark.svg"):
        assert request(http_server, path)[0] == 200
    for path in ("/.env", "/runtime.py", "/../.env", "/sessions/private.csv"):
        assert request(http_server, path)[0] == 404
    assert request(http_server, "/api/frame")[0] == 204


def test_http_rejects_untrusted_mutations(http_server):
    assert request(http_server, "/api/start", "POST", "{}")[0] == 403
    headers = {"X-GuideSense-Token": "test-token", "Origin": "https://untrusted.example"}
    assert request(http_server, "/api/start", "POST", "{}", headers)[0] == 403
    assert request(http_server, "/api/state", headers={"Host": "untrusted.example"})[0] == 403


def test_http_start_stop_and_validation(http_server, controller):
    headers = {"X-GuideSense-Token": "test-token"}
    assert request(http_server, "/api/start", "POST", "not-json", headers)[0] == 400
    assert request(http_server, "/api/start", "POST", "[]", headers)[0] == 400
    assert request(http_server, "/api/start", "POST", "x" * 4097, headers)[0] == 413
    assert request(http_server, "/api/start", "POST", "{}", headers)[0] == 200
    wait_until(lambda: controller.snapshot()["ticks"] > 0)
    assert request(http_server, "/api/start", "POST", "{}", headers)[0] == 409
    status, body, _ = request(http_server, "/api/state")
    assert status == 200 and json.loads(body)["status"] == "running"
    assert request(http_server, "/api/stop", "POST", "{}", headers)[0] == 200
    wait_until(lambda: controller.snapshot()["status"] == "stopped")
    _, exported, download_headers = request(http_server, "/api/export")
    assert b"Session ended" in exported
    assert download_headers["Content-Disposition"] == 'attachment; filename="guidesense-activity.csv"'
