"""
main.py — GuideSense Integration Entry Point
=============================================
Wires together:
  • Sensor inputs  (mock by default — swap real modules here)
  • FusionEngine   (fusion.py)
  • StateMachine   (state_machine.py)
  • HardwareInterface (stub or real GPIO/TTS)

----------------------------------------------------------------------
To switch from mock → real hardware:
  1. Replace `MockDistanceReader` with the CE freshman's serial reader.
     It must expose:  read() -> float  (metres)
  2. Replace `MockVisionReader` with the IT freshman's detector.
     It must expose:  read() -> list[Detection]
  3. Replace `HardwareInterface` stub with real TTS + GPIO buzzer class.
----------------------------------------------------------------------
"""

from __future__ import annotations

import math
import time
import random
from typing import Optional

from fusion import Detection, FusionEngine, SensorFrame
from state_machine import HardwareInterface, StateMachine


# ---------------------------------------------------------------------------
# Tick rate
# ---------------------------------------------------------------------------

TICK_HZ = 10            # updates per second
TICK_INTERVAL = 1.0 / TICK_HZ


# ---------------------------------------------------------------------------
# Mock sensor readers  (replace with real hardware below this line)
# ---------------------------------------------------------------------------

class MockDistanceReader:
    """
    Simulates a person walking toward the device and then away.

    Profile (total ~8 s at 10 Hz):
      0.0 → 3.0 s : 4.0 m  (far, silent)
      3.0 → 5.0 s : ramps 4.0 → 1.2 m  (mid, informative)
      5.0 → 6.0 s : ramps 1.2 → 0.4 m  (near, URGENT)
      6.0 → 8.0 s : ramps 0.4 → 3.5 m  (clearing, hysteresis then silent)
    """

    _PROFILE = [
        (0.0, 3.0, 4.0,  4.0),    # (t_start, t_end, d_start, d_end)
        (3.0, 5.0, 4.0,  1.2),
        (5.0, 6.0, 1.2,  0.4),
        (6.0, 8.0, 0.4,  3.5),
    ]

    def __init__(self):
        self._start = time.monotonic()

    def read(self) -> float:
        elapsed = time.monotonic() - self._start
        for t0, t1, d0, d1 in self._PROFILE:
            if elapsed <= t1:
                t = max(elapsed, t0)
                frac = (t - t0) / (t1 - t0)
                return d0 + (d1 - d0) * frac
        return 3.5   # steady far after profile ends


class MockVisionReader:
    """
    Simulates detections:
      • 'chair'  appears at 2 Hz during seconds 3–5  (mid zone)
      • 'person' appears at 10 Hz during seconds 3–8 (mid → near)
    Flicker is baked in — chair drops out every other tick to test persistence gate.
    """

    def __init__(self):
        self._start = time.monotonic()
        self._tick  = 0

    def read(self) -> list[Detection]:
        elapsed = time.monotonic() - self._start
        self._tick += 1
        detections: list[Detection] = []

        if 3.0 <= elapsed <= 8.0:
            detections.append(Detection(
                label="person",
                confidence=0.85,
                bbox=(100, 80, 400, 460),
            ))

        # Chair flickers every other tick to validate persistence gate
        if 3.0 <= elapsed <= 5.0 and self._tick % 2 == 0:
            detections.append(Detection(
                label="chair",
                confidence=0.72,
                bbox=(50, 200, 280, 460),
            ))

        return detections


# ---------------------------------------------------------------------------
# Real hardware interface stub  (replace with GPIO / pyttsx3 / etc.)
# ---------------------------------------------------------------------------

class RealHardwareInterface(HardwareInterface):
    """
    Drop-in replacement. Uncomment and fill in real hardware calls.
    """

    def speak(self, text: str) -> None:
        # e.g.:
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(text)
        # engine.runAndWait()
        print(f"[TTS]     '{text}'")    # ← replace with real TTS call

    def buzzer_on(self) -> None:
        # e.g.: GPIO.output(BUZZER_PIN, GPIO.HIGH)
        print("[BUZZER]  *** ON ***")   # ← replace with GPIO call

    def buzzer_off(self) -> None:
        # e.g.: GPIO.output(BUZZER_PIN, GPIO.LOW)
        print("[BUZZER]  --- off ---")  # ← replace with GPIO call


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    distance_reader=None,
    vision_reader=None,
    hw: Optional[HardwareInterface] = None,
    duration_sec: float = 10.0,
    verbose: bool = True,
) -> None:
    """
    Main sensing loop.

    Args:
        distance_reader : object with .read() -> float (metres).
                          Defaults to MockDistanceReader.
        vision_reader   : object with .read() -> list[Detection].
                          Defaults to MockVisionReader.
        hw              : HardwareInterface instance.
                          Defaults to RealHardwareInterface (stub).
        duration_sec    : how many seconds to run (0 = run forever).
        verbose         : print per-tick state to stdout.
    """
    distance_reader = distance_reader or MockDistanceReader()
    vision_reader   = vision_reader   or MockVisionReader()
    hw              = hw              or RealHardwareInterface()

    engine = FusionEngine()
    sm     = StateMachine(hw=hw)

    start  = time.monotonic()
    tick   = 0

    print("=" * 60)
    print("GuideSense — running")
    print("=" * 60)

    while True:
        now = time.monotonic()
        if duration_sec > 0 and (now - start) >= duration_sec:
            break

        # ------- gather sensor data -------
        distance_m = distance_reader.read()
        detections = vision_reader.read()

        frame = SensorFrame(
            distance_m=distance_m,
            detections=detections,
            timestamp=now,
        )

        # ------- fusion decision ----------
        result = engine.process(frame)

        # ------- state machine output -----
        state  = sm.update(result)

        # ------- optional verbose log -----
        if verbose:
            det_summary = ", ".join(
                f"{d.label}({d.confidence:.2f})" for d in detections
            ) or "—"
            print(
                f"t={now - start:5.2f}s  dist={distance_m:.2f}m  "
                f"dets=[{det_summary}]  "
                f"fusion={result.action.name:<12}  state={state.name}"
            )

        # ------- sleep to maintain tick rate ------
        elapsed = time.monotonic() - now
        sleep_for = max(0.0, TICK_INTERVAL - elapsed)
        time.sleep(sleep_for)
        tick += 1

    print("=" * 60)
    print("GuideSense — stopped")
    print("=" * 60)


if __name__ == "__main__":
    run(duration_sec=10.0, verbose=True)
