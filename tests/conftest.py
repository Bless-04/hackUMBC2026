"""
conftest.py — shared pytest fixtures for GuideSense test suite
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fusion import Detection, FusionEngine, SensorFrame
from state_machine import HardwareInterface, StateMachine


# ---------------------------------------------------------------------------
# Builder helpers (available to every test via import or direct call)
# ---------------------------------------------------------------------------

def make_det(label: str, confidence: float = 0.90) -> Detection:
    """Return a Detection with a dummy bounding box."""
    return Detection(label=label, confidence=confidence, bbox=(0, 0, 100, 100))


def make_frame(
    distance_m: float,
    detections: list[Detection],
    timestamp: float,
) -> SensorFrame:
    return SensorFrame(distance_m=distance_m, detections=detections, timestamp=timestamp)


def run_ticks(engine: FusionEngine, frames: list[SensorFrame]):
    """Feed frames through the engine and return the list of FusionActions."""
    return [engine.process(f).action for f in frames]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Fresh FusionEngine with standard hackathon-tuned parameters."""
    return FusionEngine(persistence_ticks=3, cooldown_sec=12.0)


@pytest.fixture
def strict_engine():
    """FusionEngine with persistence_ticks=5 — used for sudden-close tests."""
    return FusionEngine(persistence_ticks=5, cooldown_sec=12.0)


@pytest.fixture
def mock_hw():
    """Mocked HardwareInterface so tests never touch real TTS/GPIO."""
    return MagicMock(spec=HardwareInterface)


@pytest.fixture
def sm(mock_hw):
    """StateMachine wired to mock hardware."""
    return StateMachine(hw=mock_hw)


@pytest.fixture
def urgent_result():
    """A pre-built URGENT FusionResult for state machine tests."""
    from fusion import FusionAction, FusionResult, Zone
    return FusionResult(
        action=FusionAction.URGENT,
        distance_m=0.40,
        zone=Zone.NEAR,
        reason="fixture",
    )


@pytest.fixture
def silent_result():
    """A pre-built SILENT FusionResult for state machine tests."""
    from fusion import FusionAction, FusionResult, Zone
    return FusionResult(
        action=FusionAction.SILENT,
        distance_m=3.0,
        zone=Zone.FAR,
        reason="fixture",
    )


@pytest.fixture
def informative_result():
    """A pre-built INFORMATIVE FusionResult for state machine tests."""
    from fusion import FusionAction, FusionResult, Zone
    return FusionResult(
        action=FusionAction.INFORMATIVE,
        label="person",
        zone=Zone.MID,
        reason="fixture",
    )
