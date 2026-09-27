"""GuideSense command-line sensing loop for a USB camera on any desktop OS.

Default mode is simulated. Camera mode uses OpenCV detections and estimates
distance from bounding boxes. Optional speech and urgent tones use the
computer's selected audio output.
"""

from __future__ import annotations

import argparse
import time

from fusion import Detection, FusionAction, FusionEngine, SensorFrame
from state_machine import HardwareInterface, StateMachine

TICK_HZ = 10
TICK_INTERVAL = 1.0 / TICK_HZ


class CompositeHardwareInterface(HardwareInterface):
    """Connect state machine announcements to system speech and alert output."""

    def __init__(self, audio, alert) -> None:
        self._audio = audio
        self._alert = alert

    def speak(self, text: str) -> None:
        self._audio.speak(text)

    def buzzer_on(self) -> None:
        self._alert.buzzer_on()

    def buzzer_off(self) -> None:
        self._alert.buzzer_off()

    def close(self) -> None:
        if hasattr(self._alert, "cleanup"):
            self._alert.cleanup()
        if hasattr(self._audio, "close"):
            self._audio.close()


class MockDistanceReader:
    """Simulate an approach and retreat for development without a camera."""

    _PROFILE = [
        (0.0, 3.0, 4.0, 4.0),
        (3.0, 5.0, 4.0, 1.2),
        (5.0, 6.0, 1.2, 0.4),
        (6.0, 8.0, 0.4, 3.5),
    ]

    def __init__(self) -> None:
        self._start = time.monotonic()

    def read(self) -> float:
        elapsed = time.monotonic() - self._start
        for t0, t1, d0, d1 in self._PROFILE:
            if elapsed <= t1:
                fraction = (max(elapsed, t0) - t0) / (t1 - t0)
                return d0 + (d1 - d0) * fraction
        return 3.5


class MockVisionReader:
    """Create intermittent test detections for the simulated mode."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._tick = 0

    def read(self) -> list[Detection]:
        elapsed = time.monotonic() - self._start
        self._tick += 1
        detections = []
        if 3.0 <= elapsed <= 8.0:
            detections.append(Detection("person", 0.85, (100, 80, 400, 460)))
        if 3.0 <= elapsed <= 5.0 and self._tick % 2 == 0:
            detections.append(Detection("chair", 0.72, (50, 200, 280, 460)))
        return detections

    @property
    def latest_frame(self):
        return None


class MockAudioOutput:
    def speak(self, text: str) -> None:
        print(f"[TTS] {text}")


class MockHapticOutput:
    def buzzer_on(self) -> None:
        print("[ALERT] ON")

    def buzzer_off(self) -> None:
        print("[ALERT] OFF")

    def cleanup(self) -> None:
        pass


def _load_output(enable_audio: bool) -> CompositeHardwareInterface:
    if not enable_audio:
        return CompositeHardwareInterface(MockAudioOutput(), MockHapticOutput())
    from audio import AudioOutput
    from haptics import HapticOutput

    audio = AudioOutput()
    try:
        alert = HapticOutput()
    except Exception:
        audio.close()
        raise
    return CompositeHardwareInterface(audio, alert)


def run(
    use_camera: bool = False,
    enable_audio: bool = False,
    camera_index: int = 0,
    enable_gemini: bool = False,
    enable_backboard: bool = False,
    enable_logging: bool = True,
    enable_gui: bool = False,
    duration_sec: float = 10.0,
    verbose: bool = True,
) -> None:
    """Run detection and guidance; ``use_camera`` selects the live USB webcam."""
    vision_reader = output = logger = hud = None
    try:
        if use_camera:
            from distance_estimator import CameraDistanceEstimator
            from vision import VisionReader

            vision_reader = VisionReader(camera_index=camera_index)
            estimator = CameraDistanceEstimator()
            distance_reader = None
            print(f"[main] Using live camera {camera_index} and camera distance estimation")
        else:
            vision_reader = MockVisionReader()
            distance_reader = MockDistanceReader()
            estimator = None
            print("[main] Using simulated camera and distance")

        output = _load_output(enable_audio)
        if enable_logging:
            from logger import EventLogger

            logger = EventLogger()

        narrator = None
        if enable_gemini and use_camera:
            from gemini_narrator import GeminiNarrator

            narrator = GeminiNarrator()
        memory = None
        if enable_backboard and use_camera:
            from backboard_memory import BackboardMemory

            memory = BackboardMemory()

        if enable_gui:
            from hud import GuideSenseHUD

            hud = GuideSenseHUD(features={
                "Camera": f"Live {camera_index}" if use_camera else "Demo",
                "Gemini": "Active" if narrator and narrator.is_available else "Off",
                "Backboard": "Active" if memory else "Off",
                "Logging": "Active" if logger else "Off",
            })

        engine = FusionEngine()
        machine = StateMachine(output)
        start = time.monotonic()
        print("GuideSense running. Press Ctrl+C to stop.")

        while True:
            now = time.monotonic()
            if duration_sec > 0 and now - start >= duration_sec:
                break

            detections = vision_reader.read()
            image = vision_reader.latest_frame
            width = 640
            if estimator is not None:
                if image is not None:
                    height, width = image.shape[:2]
                    estimator.frame_height = height
                estimator.update_detections(detections)
                distance_m = estimator.read()
            else:
                distance_m = distance_reader.read()

            frame = SensorFrame(
                distance_m=distance_m,
                detections=detections,
                timestamp=now,
                frame_width=width,
            )
            result = engine.process(frame)
            state = machine.update(result)

            if hud is not None:
                action = hud.render(frame_img=image, frame=frame, result=result, state=state)
                if action == "QUIT":
                    break

            if narrator and result.action == FusionAction.INFORMATIVE and result.label:
                image_bytes = None
                if image is not None:
                    import cv2

                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        image_bytes = encoded.tobytes()
                narrator.describe_scene_async(
                    image_bytes=image_bytes,
                    label=result.label,
                    distance_m=distance_m,
                    on_complete=output.speak,
                )

            if memory and result.action == FusionAction.INFORMATIVE and result.label:
                memory.record_observation(label=result.label, distance_m=distance_m)
            if logger is not None:
                logger.log_event(frame, result, state)
            if verbose:
                detected = ", ".join(f"{d.label}({d.confidence:.2f})" for d in detections) or "—"
                print(f"t={now - start:5.2f}s  dist={distance_m:.2f}m  "
                      f"dets=[{detected}]  fusion={result.action.name:<12}  state={state.name}")

            time.sleep(max(0.0, TICK_INTERVAL - (time.monotonic() - now)))
    finally:
        if hud is not None:
            hud.close()
        if output is not None:
            output.close()
        if vision_reader is not None and hasattr(vision_reader, "close"):
            vision_reader.close()
        if logger is not None:
            logger.close()
        print("GuideSense stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GuideSense camera guidance")
    parser.add_argument("--camera-distance", action="store_true", help="Use live camera and camera estimated distance")
    parser.add_argument("--real-vision", action="store_true", help="Alias for --camera-distance")
    parser.add_argument("--real", action="store_true", help="Use live camera plus system speech and alert tone")
    parser.add_argument("--audio", action="store_true", help="Enable system speech and urgent tone")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (usually 0 or 1)")
    parser.add_argument("--preview", action="store_true", help="Display the live visual HUD")
    parser.add_argument("--gui", action="store_true", help="Display the live visual HUD")
    parser.add_argument("--gemini", action="store_true", help="Enable contextual Gemini narration")
    parser.add_argument("--backboard", action="store_true", help="Enable optional Backboard memory")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV event logging")
    parser.add_argument("--forever", action="store_true", help="Run until Ctrl+C")
    parser.add_argument("--duration", type=float, default=10.0, help="Run duration in seconds")
    args = parser.parse_args()

    run(
        use_camera=args.real or args.real_vision or args.camera_distance or args.gui or args.preview,
        enable_audio=args.real or args.audio,
        camera_index=args.camera,
        enable_gemini=args.gemini,
        enable_backboard=args.backboard,
        enable_logging=not args.no_log,
        enable_gui=args.gui or args.preview,
        duration_sec=0.0 if args.forever else args.duration,
    )


if __name__ == "__main__":
    main()
