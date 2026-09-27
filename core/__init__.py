"""
core — GuideSense Decision Core & Sensor Fusion
================================================
Implements real-time sensor fusion arbitration, zone gating, and system state machine.
"""

from core.fusion import (
    CONFIDENCE_MIN,
    COOLDOWN_SEC,
    MID_THRESHOLD_M,
    NEAR_THRESHOLD_M,
    OBJECT_PRIORITY,
    PERSISTENCE_TICKS,
    Detection,
    Direction,
    FusionAction,
    FusionEngine,
    FusionResult,
    SensorFrame,
    Zone,
    compute_direction,
)
from core.state_machine import (
    URGENT_HYSTERESIS_SEC,
    HardwareInterface,
    StateMachine,
    SystemState,
    format_voice_message,
)

__all__ = [
    "CONFIDENCE_MIN",
    "COOLDOWN_SEC",
    "Detection",
    "Direction",
    "FusionAction",
    "FusionEngine",
    "FusionResult",
    "HardwareInterface",
    "MID_THRESHOLD_M",
    "NEAR_THRESHOLD_M",
    "OBJECT_PRIORITY",
    "PERSISTENCE_TICKS",
    "SensorFrame",
    "StateMachine",
    "SystemState",
    "URGENT_HYSTERESIS_SEC",
    "Zone",
    "compute_direction",
    "format_voice_message",
]
