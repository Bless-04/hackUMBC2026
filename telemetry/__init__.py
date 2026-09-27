"""
telemetry — GuideSense Event Logging & Session Telemetry
========================================================
Records sensor frames, fusion decisions, and state machine transitions to CSV.
"""

from telemetry.logger import EventLogger

__all__ = [
    "EventLogger",
]
