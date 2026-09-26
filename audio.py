"""
audio.py — GuideSense Voice Alert Module (Text-to-Speech)
=========================================================
Non-blocking text-to-speech output for INFORMATIVE object announcements.

Contract:
  audio.AudioOutput().speak(text: str) -> None
    Speaks `text` aloud using offline TTS in a background thread.
    Returns immediately (< 1 ms) without stalling the main 10 Hz sensing loop.

Engines Supported:
  - macOS: Native `say` command (crisp, zero dependency)
  - Windows: SAPI.SpVoice / pyttsx3
  - Linux / Raspberry Pi: `espeak` / `spd-say` / pyttsx3
"""

from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import threading
import time
from typing import Optional


class AudioOutput:
    """
    Non-blocking, thread-safe speech synthesis engine.
    Queues spoken text and processes audio asynchronously in a worker thread.
    """

    def __init__(self) -> None:
        self._system = platform.system()
        self._engine_type = "print_only"
        self._speech_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._init_tts_engine()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _init_tts_engine(self) -> None:
        # Check available speech engines in priority order
        if self._system == "Darwin" and shutil.which("say"):
            self._engine_type = "say"
            print("[AudioOutput] macOS native 'say' TTS engine active")
            return

        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 160)
            self._engine_type = "pyttsx3"
            print("[AudioOutput] pyttsx3 TTS engine active")
            return
        except Exception:
            pass

        if shutil.which("espeak"):
            self._engine_type = "espeak"
            print("[AudioOutput] Linux 'espeak' TTS engine active")
        elif shutil.which("spd-say"):
            self._engine_type = "spd-say"
            print("[AudioOutput] Linux 'spd-say' TTS engine active")
        else:
            self._engine_type = "print_only"
            print("[AudioOutput] Audio speech synthesis ready (console mode)")

    def _worker_loop(self) -> None:
        """Background daemon worker that sequentially speaks queued words."""
        while not self._stop_event.is_set():
            try:
                text = self._speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None or self._stop_event.is_set():
                break

            try:
                if self._engine_type == "say":
                    # macOS native speech
                    subprocess.run(
                        ["say", "-r", "180", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif self._engine_type == "espeak":
                    # Linux espeak
                    subprocess.run(
                        ["espeak", "-s", "150", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif self._engine_type == "spd-say":
                    subprocess.run(
                        ["spd-say", text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                elif self._engine_type == "pyttsx3":
                    with self._lock:
                        self._pyttsx3_engine.say(text)
                        self._pyttsx3_engine.runAndWait()
            except Exception as e:
                print(f"[AudioOutput] Error speaking '{text}': {e}")
            finally:
                self._speech_queue.task_done()

    def speak(self, text: str) -> None:
        """
        Non-blocking speech trigger.
        Prints alert to console and enqueues audio speech asynchronously.
        """
        print(f"[TTS]     '{text}'")
        if self._engine_type != "print_only":
            self._speech_queue.put(text)

    def cleanup(self) -> None:
        """Shuts down the TTS worker thread cleanly."""
        self._stop_event.set()
        self._speech_queue.put(None)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    audio = AudioOutput()
    print("Testing Audio speech output...")
    for item in ["person", "chair", "bicycle"]:
        audio.speak(item)
        time.sleep(1.8)
    audio.cleanup()
    print("Audio test finished.")
