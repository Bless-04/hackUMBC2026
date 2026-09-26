"""
serial_reader.py — GuideSense Ultrasonic Serial Reader Module
==============================================================
Reads ultrasonic distance measurements streamed from the Arduino over USB serial.

Data contract:
  serial_reader.SerialDistanceReader().read() -> float (metres)

Protocol:
  Arduino transmits: "D:<distance_cm>\\n"  (e.g. "D:145\\n" -> 1.45 m)
  Pi/Host transmits: "STATE:<SILENT|INFORMATIVE|URGENT>\\n" to toggle Arduino breadboard LEDs
"""

from __future__ import annotations

import glob
import sys
import time
from typing import Optional

SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 9600
READ_TIMEOUT  = 0.08
FALLBACK_DIST = 3.5  # metres (FAR zone)


def auto_detect_serial_port() -> Optional[str]:
    """Scan and return the first available Arduino/microcontroller USB serial port."""
    patterns = [
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "COM*",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


class SerialDistanceReader:
    """
    Reads real-time distance measurements from the Arduino microcontroller.
    Auto-detects active serial ports on macOS, Linux, and Windows.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = BAUD_RATE,
        timeout: float = READ_TIMEOUT,
    ) -> None:
        self._baud = baud
        self._timeout = timeout
        self._last = FALLBACK_DIST
        self._ser = None

        target_port = port or auto_detect_serial_port() or SERIAL_PORT
        self._port = target_port
        self._init_serial(target_port)

    def _init_serial(self, port: str) -> None:
        try:
            import serial
            self._ser = serial.Serial(port, self._baud, timeout=self._timeout)
            time.sleep(1.5)  # Allow Arduino bootloader to initialize
            print(f"[SerialReader] Connected to Arduino on {port} ({self._baud} baud)")
        except ImportError:
            print("[SerialReader] pyserial not installed (pip install pyserial). Running in simulation fallback mode.")
        except Exception as e:
            print(f"[SerialReader] Serial port {port} unavailable ({e}). Using fallback.")

    def read(self) -> float:
        """
        Returns the latest distance reading in METRES.
        Falls back to 3.5m (FAR zone) if no active stream is received.
        """
        if self._ser is None or not self._ser.is_open:
            return self._last

        try:
            # Drain buffer to read the freshest line
            while self._ser.in_waiting > 32:
                self._ser.readline()

            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("D:"):
                cm_str = line[2:].strip()
                cm = float(cm_str)
                if 2.0 <= cm <= 500.0:
                    self._last = round(cm / 100.0, 2)
        except Exception:
            pass

        return self._last

    def send_state(self, state_name: str) -> None:
        """Transmits current system state to Arduino to drive Breadboard LEDs."""
        if self._ser is not None and self._ser.is_open:
            try:
                msg = f"STATE:{state_name}\n".encode("utf-8")
                self._ser.write(msg)
            except Exception:
                pass

    def close(self) -> None:
        """Closes serial connection cleanly."""
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.close()
                print("[SerialReader] Serial port closed.")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reader = SerialDistanceReader()
    print("Reading serial distance stream. Press Ctrl-C to stop.\n")
    try:
        while True:
            d = reader.read()
            print(f"Distance: {d:.2f} m", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.close()
        print("\n[SerialReader] Stopped.")
