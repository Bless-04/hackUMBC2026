"""Urgent alert output using a speaker tone, Arduino serial, or Pi GPIO."""

from __future__ import annotations

import io
import math
import platform
import shutil
import struct
import subprocess
import threading
import wave
from typing import Any

OUTPUT_MODE = "SPEAKER"
BUZZER_SERIAL_PORT = "/dev/ttyUSB0"
BUZZER_BAUD_RATE = 9600
BUZZER_GPIO_PIN = 18
TONE_HZ = 880
TONE_DURATION_SEC = 0.20
TONE_GAP_SEC = 0.08

_VALID_MODES = {"SPEAKER", "SERIAL", "GPIO"}


def _tone_wav(frequency: int, duration: float, sample_rate: int = 22050) -> bytes:
    """Create a short 16-bit mono WAV tone without third-party packages."""
    sample_count = int(sample_rate * duration)
    frames = bytearray()
    amplitude = 12_000
    for index in range(sample_count):
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        frames.extend(struct.pack("<h", sample))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)
    return buffer.getvalue()


class HapticOutput:
    """Control an idempotent urgent alert through one selected backend."""

    def __init__(
        self,
        mode: str = OUTPUT_MODE,
        *,
        serial_port: str = BUZZER_SERIAL_PORT,
        baud: int = BUZZER_BAUD_RATE,
        gpio_pin: int = BUZZER_GPIO_PIN,
        serial_connection: Any | None = None,
        gpio_module: Any | None = None,
    ) -> None:
        normalized_mode = mode.upper()
        if normalized_mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}")

        self._mode = normalized_mode
        self._active = False
        self._closed = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._serial: Any | None = None
        self._owns_serial = False
        self._gpio: Any | None = None
        self._gpio_pin = gpio_pin
        self._tone = b""
        self._audio_player: str | None = None

        if self._mode == "SPEAKER":
            if platform.system() != "Windows":
                self._audio_player = shutil.which("aplay")
                self._tone = _tone_wav(TONE_HZ, TONE_DURATION_SEC)
                if self._audio_player is None:
                    print("[HapticOutput] aplay not found; using terminal bell fallback")
            print("[HapticOutput] speaker urgency tone active")
        elif self._mode == "SERIAL":
            if serial_connection is not None:
                self._serial = serial_connection
            else:
                try:
                    import serial
                except ImportError as exc:  # pragma: no cover - host dependent
                    raise RuntimeError(
                        "pyserial is required for SERIAL haptics mode"
                    ) from exc
                self._serial = serial.Serial(serial_port, baud, timeout=0.1)
                self._owns_serial = True
            print(f"[HapticOutput] serial alert active on {serial_port}")
        else:
            if gpio_module is None:
                try:
                    import RPi.GPIO as gpio
                except ImportError as exc:  # pragma: no cover - Pi dependent
                    raise RuntimeError("RPi.GPIO is required for GPIO haptics mode") from exc
                gpio_module = gpio
            self._gpio = gpio_module
            self._gpio.setmode(self._gpio.BCM)
            self._gpio.setup(self._gpio_pin, self._gpio.OUT, initial=self._gpio.LOW)
            print(f"[HapticOutput] GPIO alert active on BCM pin {self._gpio_pin}")

    @property
    def is_active(self) -> bool:
        return self._active

    def _speaker_alarm_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if platform.system() == "Windows":
                try:
                    import winsound

                    winsound.Beep(TONE_HZ, int(TONE_DURATION_SEC * 1000))
                except (ImportError, RuntimeError):
                    stop_event.wait(TONE_DURATION_SEC)
            elif self._audio_player is not None:
                try:
                    subprocess.run(
                        [self._audio_player, "-q"],
                        input=self._tone,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=TONE_DURATION_SEC + 1.0,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    stop_event.wait(TONE_DURATION_SEC)
            else:
                print("\a", end="", flush=True)
                stop_event.wait(TONE_DURATION_SEC)

            stop_event.wait(TONE_GAP_SEC)

    def buzzer_on(self) -> None:
        """Start the urgent alert, returning immediately and doing nothing if active."""
        with self._lock:
            if self._closed:
                return
            if self._active:
                return

            if self._mode == "SPEAKER":
                # A fresh event avoids an old worker being revived by a rapid
                # off/on transition.
                self._stop_event = threading.Event()
                self._thread = threading.Thread(
                    target=self._speaker_alarm_loop,
                    args=(self._stop_event,),
                    name="guidesense-urgent-tone",
                    daemon=True,
                )
                self._thread.start()
            elif self._mode == "SERIAL":
                self._serial.write(b"BUZZ_ON\n")
            else:
                self._gpio.output(self._gpio_pin, self._gpio.HIGH)
            self._active = True

        print("[HapticOutput] *** URGENT ALARM ON ***")

    def buzzer_off(self) -> None:
        """Stop the urgent alert, returning immediately and doing nothing if inactive."""
        with self._lock:
            if not self._active:
                return

            if self._mode == "SPEAKER" and self._stop_event is not None:
                self._stop_event.set()
            elif self._mode == "SERIAL":
                self._serial.write(b"BUZZ_OFF\n")
            elif self._mode == "GPIO":
                self._gpio.output(self._gpio_pin, self._gpio.LOW)
            self._active = False

        print("[HapticOutput] --- URGENT ALARM OFF ---")

    def cleanup(self) -> None:
        """Stop output and release resources. Safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
        self.buzzer_off()

        with self._lock:
            self._closed = True
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=TONE_DURATION_SEC + TONE_GAP_SEC + 0.25)

        if self._mode == "SERIAL" and self._owns_serial and self._serial is not None:
            self._serial.close()
        elif self._mode == "GPIO" and self._gpio is not None:
            self._gpio.cleanup(self._gpio_pin)

    def __enter__(self) -> "HapticOutput":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()


if __name__ == "__main__":
    import time

    output = HapticOutput()
    try:
        print("Alarm ON for 1 second...")
        output.buzzer_on()
        time.sleep(1.0)
        print("Alarm OFF for 1 second...")
        output.buzzer_off()
        time.sleep(1.0)
        print("Alarm ON for 1 second...")
        output.buzzer_on()
        time.sleep(1.0)
    finally:
        output.cleanup()
    print("Done.")
