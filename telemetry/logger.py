"""CSV event logging for post-run review and demo evidence."""

from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import TextIO

from core.fusion import FusionResult, SensorFrame
from core.state_machine import SystemState

DEFAULT_LOG_PATH = "guidesense_log.csv"
FLUSH_EVERY_N = 10

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


class EventLogger:
    """Append one CSV record for every sensor/fusion tick."""

    def __init__(
        self,
        path: str | os.PathLike[str] = DEFAULT_LOG_PATH,
        *,
        flush_every_n: int = FLUSH_EVERY_N,
    ) -> None:
        if flush_every_n < 1:
            raise ValueError("flush_every_n must be at least 1")

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._row_count = 0
        self._start = time.monotonic()
        self._flush_every_n = flush_every_n
        self._lock = threading.Lock()
        self._closed = False

        needs_header = not self._path.exists() or self._path.stat().st_size == 0
        self._file: TextIO = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if needs_header:
            self._writer.writerow(CSV_HEADERS)
            self._file.flush()
        print(f"[Logger] appending to {self._path.resolve()}")

    def log_event(
        self,
        frame: SensorFrame,
        result: FusionResult,
        state: SystemState,
    ) -> None:
        """Write one event row and periodically flush it to durable storage."""
        elapsed = frame.timestamp - self._start if self._start else frame.timestamp

        det_parts = []
        for d in frame.detections:
            dir_str = f",{d.direction.name}" if d.direction else ""
            det_parts.append(f"{d.label}({d.confidence:.2f}{dir_str})")
        detections = ",".join(det_parts)

        dir_name = result.direction.name if result.direction else ""

        row = [
            f"{elapsed:.3f}",
            f"{frame.distance_m:.3f}",
            detections,
            result.action.name,
            result.label or "",
            dir_name,
            state.name,
            result.reason,
        ]

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot log to a closed EventLogger")
            self._writer.writerow(row)
            self._row_count += 1
            if self._row_count % self._flush_every_n == 0:
                self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file. Safe to call repeatedly."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()
        print(f"[Logger] closed - {self._row_count} events written to {self._path}")

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    from core.fusion import Detection, FusionAction, Zone

    demo_path = "guidesense_demo.csv"
    with EventLogger(demo_path) as event_logger:
        frame = SensorFrame(
            distance_m=1.5,
            detections=[Detection("person", 0.85, (0, 0, 1, 1))],
        )
        result = FusionResult(
            action=FusionAction.INFORMATIVE,
            label="person",
            distance_m=1.5,
            zone=Zone.MID,
            reason="demo",
        )
        event_logger.log_event(frame, result, SystemState.INFORMATIVE)

    print("\nLog contents:")
    print(Path(demo_path).read_text(encoding="utf-8"))
