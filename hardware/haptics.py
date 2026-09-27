"""Urgent tone through the computer's normal audio output."""

from __future__ import annotations

import io
import math
import struct
import threading
import wave

from hardware.audio_playback import play_wav_bytes

TONE_HZ = 880
TONE_DURATION_SEC = 0.20
TONE_GAP_SEC = 0.08


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
    """Keep the existing alert interface, backed only by a system audio tone."""

    def __init__(self) -> None:
        self._active = False
        self._closed = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._tone = _tone_wav(TONE_HZ, TONE_DURATION_SEC)

    @property
    def is_active(self) -> bool:
        return self._active

    def _speaker_alarm_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if not play_wav_bytes(self._tone, timeout=TONE_DURATION_SEC + 1.0):
                print("\a", end="", flush=True)
                stop_event.wait(TONE_DURATION_SEC)
            stop_event.wait(TONE_GAP_SEC)

    def buzzer_on(self) -> None:
        """Start the urgent tone without blocking the sensing loop."""
        with self._lock:
            if self._closed or self._active:
                return
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._speaker_alarm_loop,
                args=(self._stop_event,),
                name="guidesense-urgent-tone",
                daemon=True,
            )
            self._thread.start()
            self._active = True

    def buzzer_off(self) -> None:
        """Stop the tone; repeated calls are harmless."""
        with self._lock:
            if not self._active:
                return
            if self._stop_event is not None:
                self._stop_event.set()
            self._active = False

    def cleanup(self) -> None:
        """Stop output and release the worker. Safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
        self.buzzer_off()
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=TONE_DURATION_SEC + TONE_GAP_SEC + 1.25)

    def __enter__(self) -> HapticOutput:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()


if __name__ == "__main__":
    import time

    output = HapticOutput()
    try:
        output.buzzer_on()
        time.sleep(1.0)
    finally:
        output.cleanup()
