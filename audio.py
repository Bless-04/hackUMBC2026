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
    Non-blocking voice speaker.
    Prefers ElevenLabs natural voice if API key is present in .env,
    with automatic fallback to local offline pyttsx3.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._eleven = None
        self._pyttsx3_engine = None

        # 1. Try initializing ElevenLabs
        try:
            from eleven_audio import ElevenLabsVoice
            voice = ElevenLabsVoice()
            if voice.is_available:
                self._eleven = voice
                print("[AudioOutput] ElevenLabs Studio Voice ACTIVE")
        except Exception:
            pass

        # 2. Setup local pyttsx3 fallback
        try:
            import pyttsx3
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 150)
            self._pyttsx3_engine.setProperty("volume", 1.0)
        except Exception:
            pass

        if not self._eleven and not self._pyttsx3_engine:
            print("[AudioOutput] Console text fallback mode")

    def speak(self, text: str) -> None:
        """
        Speak `text` aloud non-blocking (fires background daemon thread).
        """
        def _run():
            with self._lock:
                # 1. Try ElevenLabs
                if self._eleven:
                    try:
                        success = self._eleven.speak(text)
                        if success:
                            return
                    except Exception as e:
                        print(f"[AudioOutput] ElevenLabs error, falling back: {e}")

                # 2. Try pyttsx3
                if self._pyttsx3_engine:
                    try:
                        self._pyttsx3_engine.say(text)
                        self._pyttsx3_engine.runAndWait()
                        return
                    except Exception:
                        pass

                # 3. Console print fallback
                print(f"[TTS Audio] '{text}'")

        t = threading.Thread(target=_run, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    tts = AudioOutput()
    print("Testing AudioOutput...")
    tts.speak("GuideSense initialized with Gemini, ElevenLabs, and Backboard.")
    time.sleep(3.0)
    print("Done.")
