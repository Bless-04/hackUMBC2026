"""
ui — GuideSense Dashboard & Visual HUD Overlay
===============================================
Local loopback web dashboard, real-time telemetry streaming, and OpenCV HUD overlay.
"""

from ui.hud import GuideSenseHUD, HUDTheme
from ui.runtime import DashboardHardware, SessionBusy, SessionController, validate_config

__all__ = [
    "DashboardHardware",
    "GuideSenseHUD",
    "HUDTheme",
    "SessionBusy",
    "SessionController",
    "validate_config",
]
