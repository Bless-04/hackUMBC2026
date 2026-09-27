"""Play WAV bytes through the host computer on Windows, macOS, or Linux."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile


def play_wav_bytes(wav_bytes: bytes, *, timeout: float = 5.0) -> bool:
    """Return whether the host audio player completed playback successfully."""
    if platform.system() == "Windows":
        try:
            import winsound

            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
            return True
        except (ImportError, OSError, RuntimeError):
            return False

    # afplay is built into macOS. Linux desktops commonly have paplay or aplay.
    player = next((path for name in ("afplay", "paplay", "aplay")
                   if (path := shutil.which(name))), None)
    if player is None:
        return False

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            path = output.name
            output.write(wav_bytes)
        command = [player, "-q", path] if os.path.basename(player) == "aplay" else [player, path]
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
