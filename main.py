"""
main.py — GuideSense Integration Entry Point
=============================================
Wires all three tracks together:

  [CE freshman]   serial_reader.SerialDistanceReader  → distance_m
  [IT freshman]   vision.VisionReader                → list[Detection]
  [IT freshman]   audio.AudioOutput  \
  [IT freshman]   haptics.HapticOutput > → HardwareInterface
  [IT freshman]   logger.EventLogger  /

  [Senior]        fusion.FusionEngine + state_machine.StateMachine

---------------------------------------------------------------------------
RUNNING MODES
---------------------------------------------------------------------------

  1. Mock mode (no hardware needed — runs right now):
       python -X utf8 main.py

  2. Real hardware mode (after teammates hand off their modules):
       python -X utf8 main.py --real

     The --real flag swaps MockDistanceReader → SerialDistanceReader
     and MockVisionReader → VisionReader automatically.
     No changes to fusion.py or state_machine.py needed.

  3. Partial mode (e.g. real distance but mock vision):
       python -X utf8 main.py --real-distance
       python -X utf8 main.py --real-vision

---------------------------------------------------------------------------
INTEGRATION CHECKLIST (senior fills this in at handoff time)
---------------------------------------------------------------------------
  [ ] CE freshman's serial_reader.py: read() returns float in metres
  [ ] IT freshman's vision.py:        read() returns list[Detection]
  [ ] IT freshman's audio.py:         speak() fires TTS non-blocking
  [ ] IT freshman's haptics.py:       buzzer_on()/off() agreed protocol with CE
  [ ] Serial port matches: SERIAL_PORT in serial_reader.py == haptics.py
  [ ] Label strings from vision.py match fusion.py OBJECT_PRIORITY keys
  [ ] Run --real for 30 s, verify CSV log looks correct
  [ ] Run test suite: python -X utf8 -m pytest
"""

from __future__ import annotations

import argparse
import time

from fusion import Detection, FusionAction, FusionEngine, SensorFrame
from state_machine import HardwareInterface, StateMachine

# ---------------------------------------------------------------------------
# Tick rate
# ---------------------------------------------------------------------------

TICK_HZ      = 10
TICK_INTERVAL = 1.0 / TICK_HZ


# ---------------------------------------------------------------------------
# Composite HardwareInterface — wraps audio.py + haptics.py
# ---------------------------------------------------------------------------

class CompositeHardwareInterface(HardwareInterface):
    """
    Bridges the state_machine's hardware calls to the IT freshman's modules.
    This is the only place senior code touches audio/haptics directly.
    """

    def __init__(self, audio, haptic) -> None:
        self._audio  = audio
        self._haptic = haptic

    def speak(self, text: str) -> None:
        self._audio.speak(text)

    def buzzer_on(self) -> None:
        self._haptic.buzzer_on()

    def buzzer_off(self) -> None:
        self._haptic.buzzer_off()


# ---------------------------------------------------------------------------
# Mock sensor readers (used when real hardware is not available)
# ---------------------------------------------------------------------------

class MockDistanceReader:
    """
    Simulates a person walking toward then away from the device.

    Profile (~8 s at 10 Hz):
      0–3 s   : 4.0 m (far, silent)
      3–5 s   : 4.0→1.2 m (mid, informative zone)
      5–6 s   : 1.2→0.4 m (near, URGENT)
      6–8 s   : 0.4→3.5 m (clearing, hysteresis then silent)
    """

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
                t = max(elapsed, t0)
                frac = (t - t0) / (t1 - t0)
                return d0 + (d1 - d0) * frac
        return 3.5


class MockVisionReader:
    """
    Flickering detections to exercise all fusion gates.
      person: ticks 3–8 s (continuous)
      chair:  ticks 3–5 s (every other tick — tests persistence gate)
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._tick  = 0

    def read(self) -> list[Detection]:
        elapsed = time.monotonic() - self._start
        self._tick += 1
        dets: list[Detection] = []

        if 3.0 <= elapsed <= 8.0:
            dets.append(Detection("person", 0.85, (100, 80, 400, 460)))

        if 3.0 <= elapsed <= 5.0 and self._tick % 2 == 0:
            dets.append(Detection("chair", 0.72, (50, 200, 280, 460)))

        return dets


class MockAudioOutput:
    def speak(self, text: str) -> None:
        print(f"[TTS]     '{text}'")


class MockHapticOutput:
    def buzzer_on(self)  -> None: print("[BUZZER]  *** ON ***")
    def buzzer_off(self) -> None: print("[BUZZER]  --- off ---")
    def cleanup(self)    -> None: pass


# ---------------------------------------------------------------------------
# Module loader — gracefully falls back to mocks if real module not ready
# ---------------------------------------------------------------------------

def _load_distance_reader(use_real: bool):
    if use_real:
        try:
            from serial_reader import SerialDistanceReader
            reader = SerialDistanceReader()
            print("[main] Using REAL SerialDistanceReader")
            return reader
        except Exception as e:
            print(f"[main] WARNING: SerialDistanceReader failed ({e}), falling back to mock")
    return MockDistanceReader()


def _load_vision_reader(use_real: bool):
    if use_real:
        try:
            from vision import VisionReader
            reader = VisionReader()
            print("[main] Using REAL VisionReader")
            return reader
        except Exception as e:
            print(f"[main] WARNING: VisionReader failed ({e}), falling back to mock")
    return MockVisionReader()


def _load_hardware(use_real: bool) -> HardwareInterface:
    if use_real:
        try:
            from audio import AudioOutput
            from haptics import HapticOutput
            audio  = AudioOutput()
            haptic = HapticOutput()
            print("[main] Using REAL AudioOutput + HapticOutput")
            return CompositeHardwareInterface(audio, haptic)
        except Exception as e:
            print(f"[main] WARNING: Real hardware failed ({e}), falling back to mock")
    return CompositeHardwareInterface(MockAudioOutput(), MockHapticOutput())


def _load_logger(enabled: bool):
    if enabled:
        try:
            from logger import EventLogger
            return EventLogger()
        except Exception as e:
            print(f"[main] WARNING: Logger failed ({e}), running without logging")
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    use_real_distance:   bool = False,
    use_camera_distance: bool = False,
    use_real_vision:     bool = False,
    use_real_hardware:   bool = False,
    enable_gemini:       bool = False,
    enable_backboard:    bool = False,
    enable_logging:      bool = True,
    duration_sec:        float = 10.0,
    verbose:             bool = True,
) -> None:
    """
    Main sensing loop.  All arguments default to mock/safe mode.

    Args:
        use_real_distance   : Use serial_reader.SerialDistanceReader instead of mock.
        use_camera_distance : Use camera bounding box height to estimate distance.
        use_real_vision     : Use vision.VisionReader instead of mock.
        use_real_hardware   : Use audio.AudioOutput + haptics.HapticOutput instead of mock.
        enable_gemini       : Use Google Gemini for contextual scene audio descriptions.
        enable_backboard    : Use Backboard.io for persistent spatial & session memory.
        enable_logging      : Write events to CSV via logger.EventLogger.
        duration_sec        : Seconds to run (0 = forever).
        verbose             : Print per-tick trace to stdout.
    """
    vision_reader = _load_vision_reader(use_real_vision)

    if use_camera_distance:
        from distance_estimator import CameraDistanceEstimator
        cam_dist_estimator = CameraDistanceEstimator()
        distance_reader = None
        print("[main] Using CAMERA BOUNDING-BOX DISTANCE ESTIMATOR (no physical ultrasonic needed)")
    else:
        cam_dist_estimator = None
        distance_reader = _load_distance_reader(use_real_distance)

    hw     = _load_hardware(use_real_hardware)
    logger = _load_logger(enable_logging)

    gemini_narrator = None
    if enable_gemini:
        from gemini_narrator import GeminiNarrator
        gemini_narrator = GeminiNarrator()
        if gemini_narrator.is_available:
            print("[main] Google Gemini Multimodal Scene Narrator ACTIVE")
        else:
            print("[main] WARNING: Gemini requested but API key not available")

    backboard = None
    if enable_backboard:
        from backboard_memory import BackboardMemory
        backboard = BackboardMemory()
        print("[main] Backboard.io Persistent Navigation Memory ACTIVE")

    engine = FusionEngine()
    sm     = StateMachine(hw=hw)
    start  = time.monotonic()

    print("=" * 60)
    print("GuideSense — running")
    print("=" * 60)

    try:
        while True:
            now = time.monotonic()
            if duration_sec > 0 and (now - start) >= duration_sec:
                break

            detections = vision_reader.read()

            if cam_dist_estimator is not None:
                cam_dist_estimator.update_detections(detections)
                distance_m = cam_dist_estimator.read()
            else:
                distance_m = distance_reader.read()

            frame = SensorFrame(
                distance_m=distance_m,
                detections=detections,
                timestamp=now,
            )

            result = engine.process(frame)
            state  = sm.update(result)

            # Trigger Gemini Scene Narrator asynchronously on new confirmed objects
            if gemini_narrator and result.action == FusionAction.INFORMATIVE and result.label:
                gemini_narrator.describe_scene_async(
                    label=result.label,
                    distance_m=distance_m,
                    on_complete=hw.speak,
                )

            # Record confirmed objects into Backboard persistent spatial memory
            if backboard and result.action == FusionAction.INFORMATIVE and result.label:
                backboard.record_observation(
                    label=result.label,
                    distance_m=distance_m,
                )

            if distance_reader is not None and hasattr(distance_reader, "send_state"):
                distance_reader.send_state(state.name)

            if logger:
                logger.log_event(frame, result, state)

            if verbose:
                det_str = ", ".join(
                    f"{d.label}({d.confidence:.2f})" for d in detections
                ) or "—"
                print(
                    f"t={now - start:5.2f}s  dist={distance_m:.2f}m  "
                    f"dets=[{det_str}]  "
                    f"fusion={result.action.name:<12}  state={state.name}"
                )

            elapsed = time.monotonic() - now
            time.sleep(max(0.0, TICK_INTERVAL - elapsed))

    finally:
        # Always clean up hardware and flush log on exit/error
        if hasattr(hw, '_haptic') and hasattr(hw._haptic, 'cleanup'):
            hw._haptic.cleanup()
        if logger:
            logger.close()

    print("=" * 60)
    print("GuideSense — stopped")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuideSense navigation aid")
    parser.add_argument("--real",            action="store_true", help="Use all real hardware modules")
    parser.add_argument("--camera-distance", action="store_true", help="Estimate distance using camera bounding boxes (no ultrasonic sensor)")
    parser.add_argument("--gemini",          action="store_true", help="Enable Google Gemini multimodal contextual scene description")
    parser.add_argument("--backboard",       action="store_true", help="Enable Backboard.io persistent spatial navigation memory")
    parser.add_argument("--real-distance",   action="store_true", help="Use real serial distance reader only")
    parser.add_argument("--real-vision",     action="store_true", help="Use real camera/detector only")
    parser.add_argument("--no-log",          action="store_true", help="Disable CSV event logging")
    parser.add_argument("--forever",         action="store_true", help="Run indefinitely (Ctrl-C to stop)")
    parser.add_argument("--duration",        type=float, default=10.0, help="Run duration in seconds (default 10)")
    args = parser.parse_args()

    run(
        use_real_distance   = args.real or args.real_distance,
        use_camera_distance = args.camera_distance,
        use_real_vision     = args.real or args.real_vision or args.camera_distance,
        use_real_hardware   = args.real,
        enable_gemini       = args.gemini,
        enable_backboard    = args.backboard,
        enable_logging      = not args.no_log,
        duration_sec        = 0.0 if args.forever else args.duration,
        verbose             = True,
    )
