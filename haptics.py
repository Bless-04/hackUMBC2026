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
# Configuration — agree on Option A or B with CE freshman
# ---------------------------------------------------------------------------

USE_SERIAL_BUZZER = True       # True = Option A (serial), False = Option B (GPIO)
BUZZER_SERIAL_PORT = "/dev/ttyUSB0"   # same port as serial_reader if Option A
BUZZER_GPIO_PIN    = 18               # BCM pin number if Option B


# ---------------------------------------------------------------------------
# Real implementation — IT freshman fills this in
# ---------------------------------------------------------------------------

class HapticOutput:
    """
    Controls the buzzer/vibrator.

    buzzer_on()  — start continuous buzz
    buzzer_off() — stop buzz
    Both are idempotent (safe to call when already in that state).
    """

    def __init__(self) -> None:
        self._active = False
        # Option A (serial) TODO:
        #   import serial
        #   self._ser = serial.Serial(BUZZER_SERIAL_PORT, 9600, timeout=0.05)
        #
        # Option B (GPIO) TODO:
        #   import RPi.GPIO as GPIO
        #   GPIO.setmode(GPIO.BCM)
        #   GPIO.setup(BUZZER_GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
        print("[HapticOutput] stub — buzzer not connected")

    def buzzer_on(self) -> None:
        """
        IT freshman TODO (Option A):
            if not self._active:
                self._ser.write(b"BUZZ_ON\n")
                self._active = True

        IT freshman TODO (Option B):
            import RPi.GPIO as GPIO
            if not self._active:
                GPIO.output(BUZZER_GPIO_PIN, GPIO.HIGH)
                self._active = True
        """
        if not self._active:
            print("[HapticOutput] stub buzzer_on")
            self._active = True

    def buzzer_off(self) -> None:
        """
        IT freshman TODO (Option A):
            if self._active:
                self._ser.write(b"BUZZ_OFF\n")
                self._active = False

        IT freshman TODO (Option B):
            import RPi.GPIO as GPIO
            if self._active:
                GPIO.output(BUZZER_GPIO_PIN, GPIO.LOW)
                self._active = False
        """
        if self._active:
            print("[HapticOutput] stub buzzer_off")
            self._active = False

    def cleanup(self) -> None:
        """Release GPIO/serial on shutdown."""
        self.buzzer_off()
        # Option B: GPIO.cleanup()


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
