"""
logger.py — IT Freshman's Module
===================================
Event logging for post-run review and demo evidence.

OWNED BY: Freshman #2 (Information Technology)
CALLED BY: Senior (CS) via logger.EventLogger().log_event()

Every fusion decision is logged to a CSV file.
This gives judges a printout of exactly what the system did and when.

Log format (one row per tick):
  timestamp, distance_m, detections, fusion_action, system_state

How to test standalone:
  python -X utf8 logger.py
  -> writes a short demo log to guidesense_demo.csv and prints it
"""

from __future__ import annotations

import csv
import os
import time

from fusion import FusionAction, FusionResult
from state_machine import SystemState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LOG_PATH = "guidesense_log.csv"
FLUSH_EVERY_N   = 10    # flush to disk every N rows (balance between safety and speed)

CSV_HEADERS = [
    "timestamp_s",
    "distance_m",
    "detections",          # "person(0.85,LEFT),chair(0.72,CENTER)"
    "fusion_action",       # SILENT / INFORMATIVE / URGENT
    "announced_label",     # label if INFORMATIVE, else ""
    "direction",           # LEFT / CENTER / RIGHT if available, else ""
    "system_state",        # SILENT / INFORMATIVE / APPROACHING / URGENT
    "reason",              # fusion engine's human-readable reason string
]


# ---------------------------------------------------------------------------
# IT freshman fills this in
# ---------------------------------------------------------------------------

class EventLogger:
    """
    Appends one CSV row per tick.

    Usage (called by senior's main.py):
        logger = EventLogger()
        logger.log_event(frame, result, state)
        ...
        logger.close()
    """

    def __init__(self, path: str = DEFAULT_LOG_PATH) -> None:
        self._path     = path
        self._row_count = 0
        self._start    = time.monotonic()

        # IT freshman: the open() + csv.writer below is already the full implementation.
        # No stub needed — this module is entirely Python stdlib, no hardware dependency.
        self._file   = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADERS)
        self._file.flush()
        print(f"[Logger] writing to {os.path.abspath(path)}")

    def log_event(
        self,
        frame,          # fusion.SensorFrame
        result: FusionResult,
        state: SystemState,
    ) -> None:
        """
        Write one row to the CSV.
        Called by main.py every tick — must return quickly.
        """
        elapsed = frame.timestamp - self._start if self._start else frame.timestamp

        det_parts = []
        for d in frame.detections:
            dir_str = f",{d.direction.name}" if d.direction else ""
            det_parts.append(f"{d.label}({d.confidence:.2f}{dir_str})")
        det_str = ",".join(det_parts)

        dir_name = result.direction.name if result.direction else ""

        self._writer.writerow([
            f"{elapsed:.3f}",
            f"{frame.distance_m:.3f}",
            det_str,
            result.action.name,
            result.label or "",
            dir_name,
            state.name,
            result.reason,
        ])

        self._row_count += 1
        if self._row_count % FLUSH_EVERY_N == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the log file cleanly on shutdown."""
        self._file.flush()
        self._file.close()
        print(f"[Logger] closed — {self._row_count} events written to {self._path}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fusion import Detection, FusionAction, FusionResult, SensorFrame, Zone

    logger = EventLogger("guidesense_demo.csv")

    # Write a few fake rows
    rows = [
        (3.0,   [],                                     FusionAction.SILENT,      None,       SystemState.SILENT),
        (1.5,   [Detection("person", 0.85, (0,0,1,1))], FusionAction.INFORMATIVE, "person",  SystemState.INFORMATIVE),
        (0.45,  [Detection("person", 0.85, (0,0,1,1))], FusionAction.URGENT,      None,       SystemState.URGENT),
        (2.5,   [],                                     FusionAction.SILENT,      None,       SystemState.SILENT),
    ]

    t = time.monotonic()
    for i, (dist, dets, action, label, state) in enumerate(rows):
        frame  = SensorFrame(distance_m=dist, detections=dets, timestamp=t + i * 0.1)
        result = FusionResult(action=action, label=label, zone=Zone.MID, reason="demo")
        logger.log_event(frame, result, state)

    logger.close()

    print("\nLog contents:")
    with open("guidesense_demo.csv", encoding="utf-8") as f:
        print(f.read())
