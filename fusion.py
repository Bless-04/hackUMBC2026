"""
fusion.py — GuideSense Sensor Fusion Engine
============================================
Consumes one distance reading and a list of Detections per tick.
Returns a FusionResult describing the recommended system action.

Rule priority (in order):
  1. DISTANCE ZONE computed first.
  2. NEAR  → URGENT immediately, no vision required, no persistence gating.
  3. MID   → confidence gate → persistence gate → cooldown gate → INFORMATIVE.
  4. FAR   → SILENT.
  5. Multi-object tie-break: highest priority label wins.

Data contract (shared with vision + ultrasonic teams):
  Detection(label: str, confidence: float, bbox: tuple[int,int,int,int], timestamp: float)
  distance: float  — metres
  timestamp: float — time.monotonic() seconds
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

# ---------------------------------------------------------------------------
# Spatial Direction Enum & Computation
# ---------------------------------------------------------------------------

class Direction(Enum):
    LEFT   = auto()
    CENTER = auto()
    RIGHT  = auto()


def compute_direction(
    bbox: tuple[int, int, int, int],
    frame_width: int = 640,
) -> Direction:
    """
    Computes spatial direction (LEFT, CENTER, RIGHT) of an object based on its
    bounding-box horizontal center normalized against the camera frame width.

    Normalized horizontal position:
        norm_x = x_center / frame_width

    Zones:
        LEFT:   norm_x < 0.33
        CENTER: 0.33 <= norm_x <= 0.66
        RIGHT:  norm_x > 0.66

    Works with arbitrary camera resolutions (e.g. 640x480, 1280x720, 1920x1080).
    """
    if frame_width <= 0:
        frame_width = 640

    x1, _, x2, _ = bbox
    x_center = (x1 + x2) / 2.0
    norm_x = x_center / float(frame_width)

    if norm_x < 0.33:
        return Direction.LEFT
    elif norm_x <= 0.66:
        return Direction.CENTER
    else:
        return Direction.RIGHT


# ---------------------------------------------------------------------------
# Shared data contract — agree with both freshmen on day 1
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single object detection from the vision module."""
    label: str                          # e.g. "person", "chair"
    confidence: float                   # 0.0–1.0
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) pixels
    timestamp: float = field(default_factory=time.monotonic)
    frame_width: Optional[int] = None   # original frame width in pixels (for resolution-independent direction)
    direction: Optional[Direction] = None  # Spatial direction (LEFT, CENTER, RIGHT)

    def __post_init__(self) -> None:
        if self.direction is None and self.bbox:
            fw = self.frame_width if (self.frame_width is not None and self.frame_width > 0) else 640
            self.direction = compute_direction(self.bbox, fw)


@dataclass
class SensorFrame:
    """One complete sensor snapshot delivered to the fusion engine each tick."""
    distance_m: float                   # metres, from ultrasonic or camera distance
    detections: list[Detection]         # may be empty
    timestamp: float = field(default_factory=time.monotonic)
    frame_width: int = 640              # resolution width for spatial awareness


# ---------------------------------------------------------------------------
# Zone enum
# ---------------------------------------------------------------------------

class Zone(Enum):
    NEAR = auto()   # < NEAR_THRESHOLD_M  — safety-critical
    MID  = auto()   # NEAR ≤ d < MID_THRESHOLD_M
    FAR  = auto()   # ≥ MID_THRESHOLD_M


# ---------------------------------------------------------------------------
# Fusion result
# ---------------------------------------------------------------------------

class FusionAction(Enum):
    SILENT      = auto()
    INFORMATIVE = auto()   # announce label once
    URGENT      = auto()   # trigger buzzer / voice alert


@dataclass
class FusionResult:
    action:     FusionAction
    label:      Optional[str] = None       # set when INFORMATIVE
    distance_m: Optional[float] = None
    zone:       Optional[Zone] = None
    direction:  Optional[Direction] = None # Direction.LEFT | CENTER | RIGHT
    confidence: Optional[float] = None     # confidence score of candidate detection
    reason:     str = ""                   # human-readable debug string


# ---------------------------------------------------------------------------
# Object priority table  (higher number = higher priority)
# ---------------------------------------------------------------------------

OBJECT_PRIORITY: dict[str, int] = {
    "person":       10,
    "bicycle":       8,
    "motorcycle":    8,
    "car":           7,
    "dog":           6,
    "chair":         5,
    "dining table":  4,
    "couch":         4,
    "bed":           3,
    "toilet":        3,
    # anything else defaults to 1  (see _priority())
}

def _priority(label: str) -> int:
    return OBJECT_PRIORITY.get(label.lower(), 1)


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

NEAR_THRESHOLD_M  = 0.60   # metres — URGENT below this
MID_THRESHOLD_M   = 2.00   # metres — INFORMATIVE zone upper bound
CONFIDENCE_MIN    = 0.50   # minimum confidence to consider a detection
PERSISTENCE_TICKS = 3      # consecutive ticks a label must appear before announcing
COOLDOWN_SEC      = 12.0   # seconds before the same label can be announced again


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Stateful fusion engine.  Call `process(frame)` every tick.

    State kept:
      _label_streak   — {label: deque of recent tick timestamps}
      _last_announced — {label: monotonic time of last announcement}
    """

    def __init__(
        self,
        near_threshold_m:  float = NEAR_THRESHOLD_M,
        mid_threshold_m:   float = MID_THRESHOLD_M,
        confidence_min:    float = CONFIDENCE_MIN,
        persistence_ticks: int   = PERSISTENCE_TICKS,
        cooldown_sec:      float = COOLDOWN_SEC,
    ):
        self.near_threshold_m  = near_threshold_m
        self.mid_threshold_m   = mid_threshold_m
        self.confidence_min    = confidence_min
        self.persistence_ticks = persistence_ticks
        self.cooldown_sec      = cooldown_sec

        # Per-label history: label → deque of the last N tick timestamps
        self._label_streak:   dict[str, deque] = {}
        # Per-label last-announced time
        self._last_announced: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: SensorFrame) -> FusionResult:
        """
        Core decision function.  Called once per tick with a SensorFrame.
        Returns a FusionResult — never raises.
        """
        zone = self._compute_zone(frame.distance_m)

        # ----------------------------------------------------------------
        # Rule 1 + 2: NEAR → URGENT unconditionally and immediately.
        # Do NOT pass through persistence logic — a suddenly-close object
        # (person stepping in front) must fire on the very first near tick.
        # ----------------------------------------------------------------
        if zone == Zone.NEAR:
            self._update_streaks(frame)  # keep streak state consistent
            best = None
            if frame.detections:
                best = max(frame.detections, key=lambda d: _priority(d.label.lower()))
            direction = (best.direction or compute_direction(best.bbox, frame.frame_width)) if best else None
            confidence = best.confidence if best else None
            label = best.label if best else None
            return FusionResult(
                action=FusionAction.URGENT,
                label=label,
                distance_m=frame.distance_m,
                zone=zone,
                direction=direction,
                confidence=confidence,
                reason=f"Near zone ({frame.distance_m:.2f}m < {self.near_threshold_m}m) — URGENT",
            )

        # ----------------------------------------------------------------
        # Rule 3 + 4 + 5: MID zone — vision-gated announcement
        # ----------------------------------------------------------------
        if zone == Zone.MID:
            self._update_streaks(frame)
            best = self._best_candidate(frame, frame.timestamp)
            if best is not None:
                self._last_announced[best.label] = frame.timestamp
                direction = best.direction or compute_direction(best.bbox, frame.frame_width)
                return FusionResult(
                    action=FusionAction.INFORMATIVE,
                    label=best.label,
                    distance_m=frame.distance_m,
                    zone=zone,
                    direction=direction,
                    confidence=best.confidence,
                    reason=(
                        f"Mid zone — '{best.label}' ({direction.name}) conf={best.confidence:.2f} "
                        f"persisted {self.persistence_ticks} ticks, cooldown clear"
                    ),
                )
            return FusionResult(
                action=FusionAction.SILENT,
                distance_m=frame.distance_m,
                zone=zone,
                reason="Mid zone — no object cleared persistence+cooldown gates",
            )

        # ----------------------------------------------------------------
        # FAR → SILENT
        # ----------------------------------------------------------------
        self._reset_streaks()  # objects gone, clear history
        return FusionResult(
            action=FusionAction.SILENT,
            distance_m=frame.distance_m,
            zone=zone,
            reason=f"Far zone ({frame.distance_m:.2f}m) — SILENT",
        )

    def reset(self) -> None:
        """Hard-reset all internal state (use between test scenarios)."""
        self._label_streak.clear()
        self._last_announced.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_zone(self, distance_m: float) -> Zone:
        if distance_m < self.near_threshold_m:
            return Zone.NEAR
        if distance_m <= self.mid_threshold_m:
            return Zone.MID
        return Zone.FAR

    def _update_streaks(self, frame: SensorFrame) -> None:
        """
        Update the per-label streak deques with labels seen this tick.
        Labels not seen this tick get their streak cleared.
        """
        seen_labels: set[str] = set()
        for det in frame.detections:
            if det.confidence >= self.confidence_min:
                label = det.label.lower()
                seen_labels.add(label)
                if label not in self._label_streak:
                    self._label_streak[label] = deque(maxlen=self.persistence_ticks)
                self._label_streak[label].append(frame.timestamp)

        # Drop labels not seen this tick — a gap resets persistence
        for label in list(self._label_streak.keys()):
            if label not in seen_labels:
                del self._label_streak[label]

    def _reset_streaks(self) -> None:
        """Clear all label streaks (used when entering FAR zone)."""
        self._label_streak.clear()

    def _best_candidate(
        self, frame: SensorFrame, now: float
    ) -> Optional[Detection]:
        """
        Among detections that pass confidence + persistence + cooldown gates,
        return the one with the highest priority label.
        Returns None if no detection qualifies.
        """
        candidates: list[Detection] = []

        for det in frame.detections:
            label = det.label.lower()
            if det.confidence < self.confidence_min:
                continue

            # Persistence gate: must have appeared in the last N consecutive ticks
            streak = self._label_streak.get(label)
            if streak is None or len(streak) < self.persistence_ticks:
                continue

            # Cooldown gate: not announced recently
            if label in self._last_announced:
                last = self._last_announced[label]
                if (now - last) < self.cooldown_sec:
                    continue

            candidates.append(det)

        if not candidates:
            return None

        # Rule 4: highest-priority label wins
        return max(candidates, key=lambda d: _priority(d.label.lower()))
