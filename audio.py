"""
audio.py — IT Freshman's Module
=================================
Text-to-speech output for INFORMATIVE announcements.

OWNED BY: Freshman #2 (Information Technology)
CALLED BY: Senior (CS) via state_machine.HardwareInterface.speak()

Contract:
  audio.AudioOutput().speak(text: str) -> None
    Speaks `text` aloud using TTS.
    Must be non-blocking (fire-and-forget) OR fast enough to return
    before the next tick fires (~100 ms).
    Recommended: run TTS in a background thread.

Recommended library: pyttsx3 (offline, no API key needed)
  pip install pyttsx3
  On Raspberry Pi also: sudo apt install espeak

How to test standalone:
  python -X utf8 audio.py
  -> speaks "person" and "chair" with a pause between
"""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# Real implementation — IT freshman fills this in
# ---------------------------------------------------------------------------

class AudioOutput:
    """
    Non-blocking TTS speaker.

    The senior calls speak() from the main loop — it must return quickly.
    Use a daemon thread so it doesn't block the tick rate.
    """

    def __init__(self) -> None:
        # IT freshman: initialise pyttsx3 engine here
        # import pyttsx3
        # self._engine = pyttsx3.init()
        # self._engine.setProperty("rate", 150)   # words per minute
        # self._engine.setProperty("volume", 1.0)
        self._lock = threading.Lock()
        print("[AudioOutput] stub — TTS not connected")

    def speak(self, text: str) -> None:
        """
        Speak `text` aloud.  Returns immediately (fires background thread).

        IT freshman TODO:
            def _run():
                with self._lock:               # prevent overlap
                    self._engine.say(text)
                    self._engine.runAndWait()
            t = threading.Thread(target=_run, daemon=True)
            t.start()
        """
        print(f"[AudioOutput] stub speak: '{text}'")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    tts = AudioOutput()
    for word in ["person", "chair", "bicycle"]:
        print(f"Speaking: {word}")
        tts.speak(word)
        time.sleep(2.5)
