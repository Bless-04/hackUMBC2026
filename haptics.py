"""
haptics.py — IT Freshman's Module
====================================
Buzzer/vibration control for URGENT alerts.

OWNED BY: Freshman #2 (Information Technology)
CALLED BY: Senior (CS) via state_machine.HardwareInterface.buzzer_on/off()

IMPORTANT — coordination with CE freshman:
  The Arduino controls the buzzer electrically, but the Pi tells it when
  to fire by sending a serial command OR by toggling a GPIO pin.
  Decide with CE freshman which method you're using and match the Arduino sketch.

  Option A (recommended — simpler): Pi sends "BUZZ_ON\n" / "BUZZ_OFF\n"
    over the SAME serial connection as distance readings.
    CE freshman adds a listener in the Arduino sketch.

  Option B: Pi toggles a dedicated GPIO pin → opto-isolator → buzzer circuit.
    Requires an extra wire but keeps audio and sensor serial separate.

Contract:
  haptics.HapticOutput().buzzer_on()  -> None
  haptics.HapticOutput().buzzer_off() -> None
  Both must return immediately (the state machine calls them from the main loop).

How to test standalone:
  python -X utf8 haptics.py
  -> buzzes for 1 s, off for 1 s, buzzes for 1 s
"""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# Configuration — choose output mode
# ---------------------------------------------------------------------------

OUTPUT_MODE = "SPEAKER"        # "SPEAKER" (JBL audio tone), "SERIAL" (Arduino), or "GPIO"
BUZZER_SERIAL_PORT = "/dev/ttyUSB0"   # for "SERIAL"
BUZZER_GPIO_PIN    = 18               # for "GPIO"


# ---------------------------------------------------------------------------
# Real implementation — IT freshman fills this in
# ---------------------------------------------------------------------------

class HapticOutput:
    """
    Controls urgent alerting via JBL speaker audio tone, Arduino serial, or Pi GPIO.
    buzzer_on()  — start continuous/pulsing urgent alert
    buzzer_off() — stop alert
    Both are idempotent.
    """

    def __init__(self, mode: str = OUTPUT_MODE) -> None:
        self._mode = mode
        self._active = False
        self._thread = None
        self._stop_event = None

        if self._mode == "SPEAKER":
            import threading
            self._stop_event = threading.Event()
            print("[HapticOutput] JBL Speaker alarm tone mode active")
        else:
            print(f"[HapticOutput] stub — mode={self._mode} not connected")

    def _speaker_alarm_loop(self) -> None:
        """Looping urgency tone played through system audio / JBL speaker."""
        import platform
        is_windows = platform.system() == "Windows"

        while not self._stop_event.is_set():
            if is_windows:
                try:
                    import winsound
                    winsound.Beep(880, 200)  # 880 Hz beep for 200ms
                except Exception:
                    time.sleep(0.2)
            else:
                # Linux / Raspberry Pi: beep via terminal bell or audio system
                import os
                # Generates a quick beep or audio cue
                os.system("(speaker-test -t sine -f 880 -l 1 > /dev/null 2>&1) &")
                time.sleep(0.25)
            time.sleep(0.05)

    def buzzer_on(self) -> None:
        if not self._active:
            self._active = True
            print("[HapticOutput] *** URGENT ALARM ON ***")
            if self._mode == "SPEAKER":
                import threading
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._speaker_alarm_loop, daemon=True)
                self._thread.start()

    def buzzer_off(self) -> None:
        if self._active:
            self._active = False
            print("[HapticOutput] --- URGENT ALARM OFF ---")
            if self._mode == "SPEAKER" and self._stop_event:
                self._stop_event.set()

    def cleanup(self) -> None:
        """Release audio/GPIO/serial on shutdown."""
        self.buzzer_off()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    h = HapticOutput()
    print("Buzzer ON for 1 s...")
    h.buzzer_on()
    time.sleep(1.0)
    print("Buzzer OFF for 1 s...")
    h.buzzer_off()
    time.sleep(1.0)
    print("Buzzer ON for 1 s...")
    h.buzzer_on()
    time.sleep(1.0)
    h.cleanup()
    print("Done.")
