"""Non-blocking text-to-speech output for GuideSense announcements."""

from __future__ import annotations

import queue
import threading
from types import ModuleType
from typing import Any, Optional

from core.fusion import Direction, FusionAction, FusionResult

# ---------------------------------------------------------------------------
# Centralized Natural-Language Voice Formatting
# ---------------------------------------------------------------------------

def format_voice_message(
    label: Optional[str] = None,
    direction: Optional[Direction] = None,
    action: FusionAction = FusionAction.INFORMATIVE,
    result: Optional[FusionResult] = None,
) -> str:
    """
    Centralized function converting a confirmed fusion event / detection into
    natural-language speech with spatial awareness.

    Examples:
      - INFORMATIVE + LEFT   -> "Person on your left."
      - INFORMATIVE + CENTER -> "Chair in front of you."
      - INFORMATIVE + RIGHT  -> "Vehicle on your right."
      - INFORMATIVE + None   -> "person"
      - URGENT + LEFT        -> "Stop. Person on your left."
      - URGENT + None        -> "Stop. Obstacle ahead."
    """
    if result is not None:
        label = result.label if label is None else label
        direction = result.direction if direction is None else direction
        action = result.action if action is None else action

    if action == FusionAction.URGENT:
        if label:
            cap_label = label.strip().capitalize()
            if direction == Direction.LEFT:
                return f"Stop. {cap_label} on your left."
            elif direction == Direction.RIGHT:
                return f"Stop. {cap_label} on your right."
            elif direction == Direction.CENTER:
                return f"Stop. {cap_label} in front of you."
            else:
                return f"Stop. {cap_label} ahead."
        else:
            if direction == Direction.LEFT:
                return "Stop. Obstacle on your left."
            elif direction == Direction.RIGHT:
                return "Stop. Obstacle on your right."
            else:
                return "Stop. Obstacle ahead."

    # INFORMATIVE path
    if not label:
        return ""

    cap_label = label.strip().capitalize()
    if direction == Direction.LEFT:
        return f"{cap_label} on your left."
    elif direction == Direction.RIGHT:
        return f"{cap_label} on your right."
    elif direction == Direction.CENTER:
        return f"{cap_label} in front of you."
    else:
        return label.strip()


# ---------------------------------------------------------------------------
# AudioOutput
# ---------------------------------------------------------------------------

class AudioOutput:
    """
    Serialize speech on one background worker so sensing is never blocked.

    ElevenLabs is preferred when configured. If it is unavailable or a call
    fails, the same utterance falls back to local ``pyttsx3`` and finally to a
    console message. A single worker avoids overlapping announcements and the
    thread-affinity issues some platform TTS engines have.
    """

    def __init__(self, *, rate: int = 150, volume: float = 1.0) -> None:
        self._rate = rate
        self._volume = volume
        self._eleven: Any | None = None
        self._pyttsx3: ModuleType | None = None
        self._pyttsx3_engine: Any | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False

        try:
            import eleven_audio

            voice_cls = getattr(eleven_audio, "ElevenLabsVoice", None)
            if callable(voice_cls):
                voice = voice_cls()
                if getattr(voice, "is_available", False):
                    self._eleven = voice
        except Exception as exc:
            print(f"[AudioOutput] ElevenLabs unavailable: {exc}")

        try:
            import pyttsx3

            self._pyttsx3 = pyttsx3
        except (ImportError, RuntimeError):
            self._pyttsx3 = None

        if self._eleven is not None:
            backend = "ElevenLabs with local fallback"
        elif self._pyttsx3 is not None:
            backend = "local pyttsx3"
        else:
            backend = "console fallback"
        print(f"[AudioOutput] {backend} active")

        self._worker = threading.Thread(
            target=self._run,
            name="guidesense-tts",
            daemon=True,
        )
        self._worker.start()

    def speak(self, text: str) -> None:
        """Queue an utterance and return immediately."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        utterance = text.strip()
        if not utterance:
            return

        with self._state_lock:
            if self._closed:
                return
            self._queue.put_nowait(utterance)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                self._speak_now(text)
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        if self._eleven is not None:
            try:
                if self._eleven.speak(text):
                    return
            except Exception as exc:
                print(f"[AudioOutput] ElevenLabs failed; using local voice: {exc}")

        if self._pyttsx3 is not None:
            try:
                # Initialize and use the engine on the same worker thread.
                if self._pyttsx3_engine is None:
                    self._pyttsx3_engine = self._pyttsx3.init()
                    self._pyttsx3_engine.setProperty("rate", self._rate)
                    self._pyttsx3_engine.setProperty("volume", self._volume)
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return
            except Exception as exc:
                print(f"[AudioOutput] local TTS failed: {exc}")
                self._pyttsx3_engine = None

        print(f"[TTS Audio] '{text}'")

    def close(self, timeout: float = 2.0) -> None:
        """Stop after already-queued speech. Safe to call repeatedly."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put_nowait(None)
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=max(0.0, timeout))

    def __enter__(self) -> "AudioOutput":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    output = AudioOutput()
    print("Testing AudioOutput...")
    output.speak("GuideSense initialized with Gemini, ElevenLabs, and Backboard.")
    output.close(timeout=10.0)
    print("Done.")
