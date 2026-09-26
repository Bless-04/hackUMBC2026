"""
serial_reader.py — CE Freshman's Module
========================================
Reads ultrasonic distance over USB serial from the Arduino.

OWNED BY: Freshman #1 (Computer Engineering)
HANDED OFF TO: Senior (CS) via the read() interface below.

Data contract with senior:
  serial_reader.SerialDistanceReader().read() -> float   (metres)

Serial protocol (Arduino sends one line per reading):
  "D:<distance_cm>\n"
  e.g.  "D:143\n"  → 1.43 m

Arduino sketch responsibilities (also owned by CE freshman):
  - HC-SR04 trigger on GPIO pin (see Part 3 Path A for wiring)
  - Reads distance, sends "D:<cm>\\n" at ~20 Hz over USB serial
  - Buzzer pin toggled by a separate signal from the Pi (see haptics.py)

How to test standalone (before handoff):
  python -X utf8 serial_reader.py
  → prints live distance readings to stdout
"""

from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# Configuration — CE freshman sets these to match their Arduino sketch
# ---------------------------------------------------------------------------

SERIAL_PORT   = "/dev/ttyUSB0"   # adjust: /dev/ttyACM0 on some Pi setups
BAUD_RATE     = 9600
READ_TIMEOUT  = 0.1              # seconds before giving up on a line
FALLBACK_DIST = 3.0              # metres — returned if serial read fails


# ---------------------------------------------------------------------------
# Real implementation — CE freshman fills this in
# ---------------------------------------------------------------------------

class SerialDistanceReader:
    """
    Reads one distance value from the Arduino over USB serial.

    Usage:
        reader = SerialDistanceReader()
        metres = reader.read()   # call this every tick

    The senior's main.py calls reader.read() at TICK_HZ (10 Hz).
    The Arduino sends at ~20 Hz — this just grabs the latest line.
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = BAUD_RATE,
        timeout: float = READ_TIMEOUT,
    ) -> None:
        # CE freshman: uncomment and fill in when hardware is ready
        # import serial
        # self._ser = serial.Serial(port, baud, timeout=timeout)
        # time.sleep(2)   # let Arduino reset after serial connect
        self._port    = port
        self._baud    = baud
        self._timeout = timeout
        self._last    = FALLBACK_DIST
        print(f"[SerialReader] stub — port={port} baud={baud}  (not connected)")

    def read(self) -> float:
        """
        Return the most recent distance in METRES.
        Returns FALLBACK_DIST on any read error so the system degrades gracefully.

        CE freshman TODO:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("D:"):
                cm = float(line[2:])
                self._last = cm / 100.0
            return self._last
        """
        # Stub: return fallback until real hardware is wired
        return self._last

    def send_state(self, state_name: str) -> None:
        """
        Sends the current SystemState to the Arduino to control Breadboard LEDs.
        (Green = SILENT, Yellow = INFORMATIVE, Red = URGENT)
        """
        # CE freshman TODO:
        #   if hasattr(self, "_ser") and self._ser and self._ser.is_open:
        #       self._ser.write(f"STATE:{state_name}\n".encode("utf-8"))
        pass

    def close(self) -> None:
        """Call on shutdown to release the serial port."""
        # if hasattr(self, "_ser") and self._ser:
        #     self._ser.close()
        pass


# ---------------------------------------------------------------------------
# Standalone test — run directly to verify serial comms before handoff
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reader = SerialDistanceReader()
    print("Reading distance. Ctrl-C to stop.")
    try:
        while True:
            dist = reader.read()
            print(f"  {dist:.3f} m")
            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.close()
        print("Done.")
