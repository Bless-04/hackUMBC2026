"""
eleven_audio.py — Ultra-Realistic Voice Output via ElevenLabs API
==================================================================
Provides human-like, natural voice output for GuideSense via ElevenLabs.
Outputs speech through the connected JBL speaker.

Features:
  - Uses the ultra-low latency model: `eleven_flash_v2_5`.
  - Uses the pre-made voice ID: `JBFqnCBsd6RMkjVDRZzb` (George).
  - Audio Format: WAV (PCM 24kHz) for zero-dependency native playback:
      - Windows: native `winsound.PlaySound`
      - Linux / Raspberry Pi: native `aplay`
  - Fallback: Gracefully falls back to local pyttsx3 or terminal log if offline or rate limited.
  - Non-blocking: Generates and plays audio in a background daemon thread so it
    NEVER stalls the 10 Hz local safety loop.
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import wave
from typing import Callable, Optional

# ElevenLabs configuration
DEFAULT_VOICE_ID = "XrExE9yKIg1WjnnlVkGX"  # Maltida
MODEL_ID = "eleven_flash_v2_5"             # Fast, low-latency model
OUTPUT_FORMAT = "pcm_24000"                # Raw PCM 24kHz, 16-bit mono


def load_elevenlabs_key() -> Optional[str]:
    """Load ELEVEN_LABS_API_KEY from environment or .env file."""
    for env_var in ["ELEVEN_LABS_API_KEY", "ELEVENLABS_API_KEY"]:
        if env_var in os.environ and os.environ[env_var].strip():
            return os.environ[env_var].strip()

    # Search in .env files
    for path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ELEVEN_LABS_API_KEY=") or line.startswith("ELEVENLABS_API_KEY="):
                        return line.split("=", 1)[1].strip("\"' ")
    return None


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wraps raw PCM 16-bit mono audio with a standard RIFF/WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit = 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


class ElevenLabsVoice:
    """
    Synthesizes and speaks text using ElevenLabs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = DEFAULT_VOICE_ID,
    ):
        self.api_key = load_elevenlabs_key() if api_key is None else api_key
        self.voice_id = voice_id
        self._lock = threading.Lock()

        if self.api_key:
            print(f"[ElevenLabsVoice] Initialized with key ({self.api_key[:8]}...) voice={voice_id}")
        else:
            print("[ElevenLabsVoice] WARNING: No ELEVEN_LABS_API_KEY found")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_wav(self, text: str) -> Optional[bytes]:
        """
        Calls ElevenLabs TTS and returns WAV bytes.
        Returns None on error.
        """
        if not self.api_key or not text.strip():
            return None

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?output_format={OUTPUT_FORMAT}"
        payload = json.dumps({
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                pcm_data = resp.read()
                return pcm_to_wav_bytes(pcm_data, sample_rate=24000)
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"[ElevenLabsVoice] HTTP {e.code}: {err_msg[:120]}")
        except Exception as e:
            print(f"[ElevenLabsVoice] API error: {e}")

        return None

    def play_wav(self, wav_bytes: bytes) -> None:
        """
        Plays WAV audio through system output (JBL speaker).
        Uses native OS utilities (winsound on Windows, aplay on Linux/Pi).
        """
        is_windows = platform.system() == "Windows"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            if is_windows:
                try:
                    import winsound
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                except Exception as e:
                    print(f"[ElevenLabsVoice] Windows audio playback error: {e}")
            else:
                # Linux / Raspberry Pi: native ALSA player
                subprocess.run(["aplay", "-q", tmp_path], check=False)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def speak(self, text: str) -> bool:
        """
        Synchronously generates and plays audio for text.
        """
        wav_bytes = self.generate_wav(text)
        if wav_bytes:
            self.play_wav(wav_bytes)
            return True
        return False

    def speak_async(self, text: str, on_done: Optional[Callable[[], None]] = None) -> bool:
        """
        Non-blocking: Generates and plays ElevenLabs voice in a background daemon thread.
        """
        if not self.is_available:
            return False

        def _worker():
            try:
                success = self.speak(text)
                if success and on_done:
                    on_done()
            except Exception as e:
                print(f"[ElevenLabsVoice] Async worker error: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True


# ---------------------------------------------------------------------------
# Standalone CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    voice = ElevenLabsVoice()
    if not voice.is_available:
        print("Please set ELEVEN_LABS_API_KEY in .env before running.")
    else:
        print("Speaking via ElevenLabs through JBL speaker...")
        voice.speak("Hello! I am GuideSense, powered by Gemini and ElevenLabs.")
        print("Done.")
