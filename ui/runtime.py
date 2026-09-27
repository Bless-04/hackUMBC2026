"""Threaded dashboard sessions using the existing GuideSense detection modules.

Only this worker touches the camera. Browser requests read small snapshots;
they never run inference or cloud requests on the HTTP thread.
"""

from __future__ import annotations

import copy
import csv
import io
import math
import threading
import time
from collections import deque
from pathlib import Path

from core.fusion import Detection, FusionAction, FusionEngine, SensorFrame
from core.state_machine import HardwareInterface, StateMachine

SESSION_DIR = Path(__file__).parent / "sessions"


def validate_config(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Session settings must be an object.")
    mode = data.get("mode", "demo")
    camera = data.get("camera", 0)
    if mode not in ("demo", "camera"):
        raise ValueError("Choose Demo or Live camera.")
    if type(camera) is not int or not 0 <= camera <= 10:
        raise ValueError("Camera index must be a whole number from 0 to 10.")
    result = {"mode": mode, "camera": camera}
    for name in ("voice", "gemini", "backboard", "logging"):
        value = data.get(name, False)
        if type(value) is not bool:
            raise ValueError(f"{name} must be true or false.")
        # Demo is local, silent, and never sends synthetic observations to cloud services.
        result[name] = value if mode == "camera" or name == "logging" else False
    return result


class SessionBusy(RuntimeError):
    pass


class DashboardHardware(HardwareInterface):
    def __init__(self, controller, generation, audio=None, haptic=None):
        self.controller = controller
        self.generation = generation
        self.audio = audio
        self.haptic = haptic

    def speak(self, text):
        self.announce(text, "Guidance")

    def announce(self, text, source):
        with self.controller.lock:
            if (self.generation != self.controller.generation
                    or self.controller.stop_event.is_set()):
                return
            self.controller.data["announcement"] = {"text": text, "source": source}
            self.controller.event(source, text)
            if self.audio is not None:
                self.audio.speak(text)

    def buzzer_on(self):
        if self.haptic is not None:
            self.haptic.buzzer_on()

    def buzzer_off(self):
        if self.haptic is not None:
            self.haptic.buzzer_off()


class SessionController:
    def __init__(self):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.generation = 0
        self.jpeg = None
        self.events = deque(maxlen=200)
        self.next_event = 0
        self.data = self._initial()

    @staticmethod
    def _initial():
        return {
            "status": "idle", "mode": "demo", "state": "SILENT", "zone": None,
            "distance": None, "fps": 0, "elapsed": 0, "detections": [],
            "frame_width": 640, "frame_height": 480, "has_frame": False,
            "announcement": None, "error": None, "ticks": 0,
            "services": {"gemini": "Off", "voice": "Off", "backboard": "Off",
                         "logging": "Off", "camera": "Standby"},
        }

    def event(self, category, message):
        with self.lock:
            self.next_event += 1
            self.events.appendleft({
                "id": self.next_event, "time": time.strftime("%H:%M:%S"),
                "category": category, "message": message,
            })

    def snapshot(self):
        with self.lock:
            return {**copy.deepcopy(self.data), "events": list(self.events)}

    def start(self, settings):
        config = validate_config(settings)
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise SessionBusy("A session is already running or stopping.")
            self.generation += 1
            self.stop_event = threading.Event()
            self.events.clear()
            self.jpeg = None
            self.data = {**self._initial(), "status": "starting", "mode": config["mode"]}
            self.event("Session", "Preparing " + ("live camera." if config["mode"] == "camera"
                                                    else "local demonstration."))
            self.thread = threading.Thread(target=self._run, args=(config, self.generation),
                                           name="guidesense-dashboard", daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                self.stop_event.set()
                self.data["status"] = "stopping"

    def shutdown(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def export(self):
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["time", "category", "message"])
        for row in reversed(self.snapshot()["events"]):
            # Prevent spreadsheet formula interpretation in downloaded event text.
            writer.writerow([("'" + str(row[key])) if str(row[key]).startswith(("=", "+", "-", "@"))
                             else row[key] for key in ("time", "category", "message")])
        return output.getvalue().encode("utf-8-sig")

    @staticmethod
    def demo_frame(elapsed):
        """Repeat a person approaching and clearing the camera every 18 seconds."""
        phase = elapsed % 18
        distance = 3.5 if phase < 2 or phase > 14 else (
            3.5 - (phase - 2) * .46 if phase < 9 else .28 + (phase - 9) * .65)
        detections = []
        if 2 <= phase <= 14:
            height = int(100 + (3.5 - distance) * 105)
            center = int(320 + math.sin(elapsed * .25) * 140)
            detections.append(Detection("person", .94, (center - 55, 455 - height,
                                                        center + 55, 455), frame_width=640))
        return max(.3, distance), detections

    def _run(self, config, generation):
        reader = audio = haptic = logger = narrator = memory = None
        failed = False
        try:
            services = dict(self.data["services"])
            if config["mode"] == "camera":
                import distance_estimator

                import vision

                reader = vision.VisionReader(camera_index=config["camera"])
                estimator = distance_estimator.CameraDistanceEstimator()
                services["camera"] = f"Camera {config['camera']}"
            else:
                services["camera"] = "Simulated"
            if config["voice"]:
                import audio as audio_mod
                import haptics as haptics_mod

                audio = audio_mod.AudioOutput()
                haptic = haptics_mod.HapticOutput()
                services["voice"] = "Enabled"
            if config["gemini"]:
                import gemini_narrator

                narrator = gemini_narrator.GeminiNarrator()
                services["gemini"] = "Configured" if narrator.is_available else "Key missing"
            if config["backboard"]:
                import backboard_memory

                memory = backboard_memory.BackboardMemory()
                services["backboard"] = "Configured" if memory.is_cloud_enabled else "Local memory"
            if config["logging"]:
                import logger as logger_mod

                path = SESSION_DIR / f"session-{time.time_ns()}.csv"
                logger = logger_mod.EventLogger(path)
                services["logging"] = "Recording"
            hardware = DashboardHardware(self, generation, audio, haptic)
            machine = StateMachine(hardware)
            engine = FusionEngine()
            import hud as hud_mod

            hud_obj = hud_mod.GuideSenseHUD(
                features={
                    "Camera": services.get("camera", "Active"),
                    "Gemini": services.get("gemini", "Off"),
                    "Backboard": services.get("backboard", "Off"),
                    "Logging": services.get("logging", "Off"),
                }
            )
            start = last_frame_time = time.monotonic()
            previous_state = None
            previous_image = None
            with self.lock:
                self.data.update(services=services, status="running")
            self.event("Session", "Camera session started." if reader else
                       "Demo started. Objects and distances are simulated; cloud and sound are off.")
            while not self.stop_event.is_set():
                tick = time.monotonic()
                elapsed = tick - start
                image = None
                width, height = 640, 480
                fresh = True
                if reader is not None:
                    detections = reader.read()
                    image = reader.latest_frame
                    fresh = image is not None and image is not previous_image
                    previous_image = image
                    if image is not None:
                        height, width = image.shape[:2]
                        estimator.frame_height = height
                    estimator.update_detections(detections)
                    distance = estimator.read()
                else:
                    distance, detections = self.demo_frame(elapsed)
                frame = SensorFrame(distance, detections, frame_width=width)
                result = engine.process(frame)
                state = machine.update(result)
                if previous_state != state.name:
                    self.event("State", f"{state.name.title()} · {result.reason}")
                    previous_state = state.name
                jpeg = None
                # In demo mode, fresh is always True, so we always produce a frame.
                # In camera mode, we only render when we have a new camera frame (fresh=True).
                # We also render on a timer even when fresh=False in demo mode so the HUD
                # animates smoothly (pulse, state changes) at ~7 FPS.
                should_render = (
                    fresh and image is not None  # camera: new raw frame arrived
                ) or (
                    reader is None  # demo: always re-render from synthetic canvas
                    and tick - last_frame_time >= .15
                )
                if should_render:
                    import cv2
                    hud_base = image if image is not None else hud_obj._create_synthetic_canvas(width, height)
                    hud_frame = hud_obj.draw_hud(hud_base, frame, result, state)
                    ok, encoded = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        jpeg = encoded.tobytes()
                    last_frame_time = tick
                if result.action == FusionAction.INFORMATIVE and result.label:
                    if narrator is not None and narrator.is_available:
                        narrator.describe_scene_async(
                            image_bytes=jpeg or self.jpeg, label=result.label, distance_m=distance,
                            on_complete=lambda text: hardware.announce(text, "Gemini"))
                    if memory is not None:
                        memory.record_observation(label=result.label, distance_m=distance)
                if logger is not None:
                    logger.log_event(frame, result, state)
                with self.lock:
                    if jpeg is not None:
                        self.jpeg = jpeg
                    elif reader is not None and not fresh:
                        # Camera mode: frame was stale/repeated — clear the stale jpeg
                        # so the UI shows a waiting indicator instead of a frozen old frame.
                        self.jpeg = None
                    self.data.update(
                        elapsed=elapsed, state=state.name, zone=result.zone.name if fresh else None,
                        distance=distance if fresh else None, detections=[{
                            "label": d.label, "confidence": d.confidence, "bbox": d.bbox,
                            "direction": d.direction.name if d.direction else "CENTER",
                        } for d in detections], frame_width=width, frame_height=height,
                        has_frame=self.jpeg is not None,
                    )
                    self.data["ticks"] += 1
                    self.data["fps"] = round(self.data["ticks"] / max(.1, elapsed + .1), 1)
                self.stop_event.wait(max(0, .1 - (time.monotonic() - tick)))
        except Exception as exc:
            failed = True
            # Raw exceptions stay on the local console; API responses never echo credentials.
            print(f"[Dashboard] Session failed: {type(exc).__name__}: {exc}")
            with self.lock:
                self.data["error"] = "Session could not continue. Check the camera, model files, and terminal output."
            self.event("Error", "Session failed. Check the terminal for details.")
        finally:
            self.stop_event.set()
            for resource, method in ((haptic, "cleanup"), (audio, "close"),
                                     (reader, "close"), (logger, "close")):
                if resource is not None:
                    try:
                        getattr(resource, method)()
                    except Exception as exc:
                        print(f"[Dashboard] Cleanup failed: {type(exc).__name__}")
            with self.lock:
                self.data.update(status="error" if failed else "stopped", has_frame=False)
                self.jpeg = None
            if not failed:
                self.event("Session", "Session ended. Camera and alert output released.")
