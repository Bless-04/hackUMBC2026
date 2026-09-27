"""
_compat.py — Backward-compatibility aliases for legacy top-level imports.

Allows code, tests, and mock runners that import e.g. `import fusion` or
`from vision import VisionReader` to resolve seamlessly to the organized domain packages.
"""

from __future__ import annotations

import sys


def register_compat_aliases() -> None:
    """Register domain packages under legacy top-level module names in sys.modules."""
    import core.fusion as _fusion
    import core.state_machine as _state_machine
    import hardware.audio as _audio
    import hardware.audio_playback as _audio_playback
    import hardware.eleven_audio as _eleven_audio
    import hardware.haptics as _haptics
    import services.backboard_memory as _backboard_memory
    import services.gemini_narrator as _gemini_narrator
    import telemetry.logger as _logger
    import ui.hud as _hud
    import vision.distance as _distance
    import vision.distance_estimator as _distance_estimator
    import vision.vision as _vision

    aliases = {
        "fusion": _fusion,
        "state_machine": _state_machine,
        "vision": _vision,
        "distance": _distance,
        "distance_estimator": _distance_estimator,
        "audio": _audio,
        "audio_playback": _audio_playback,
        "eleven_audio": _eleven_audio,
        "haptics": _haptics,
        "gemini_narrator": _gemini_narrator,
        "backboard_memory": _backboard_memory,
        "logger": _logger,
        "hud": _hud,
    }

    for name, module in aliases.items():
        sys.modules.setdefault(name, module)

register_compat_aliases()
