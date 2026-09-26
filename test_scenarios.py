"""
test_scenarios.py — GuideSense Nine-Scenario Validation Suite
==============================================================
Run with:  python test_scenarios.py
All nine scenarios from the brief are exercised here.
No external dependencies beyond the standard library.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from fusion import (
    Detection,
    FusionAction,
    FusionEngine,
    SensorFrame,
)
from state_machine import HardwareInterface, StateMachine, SystemState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(label: str, confidence: float = 0.90) -> Detection:
    return Detection(label=label, confidence=confidence, bbox=(0, 0, 100, 100))


def make_frame(distance_m: float, detections: list[Detection], timestamp: float) -> SensorFrame:
    return SensorFrame(distance_m=distance_m, detections=detections, timestamp=timestamp)


def run_ticks(engine: FusionEngine, frames: list[SensorFrame]) -> list[FusionAction]:
    """Feed a list of frames and return the list of resulting actions."""
    return [engine.process(f).action for f in frames]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScenario1_PersonAtMidRange(unittest.TestCase):
    """Person at 2 m → speak once, then silent."""

    def test_person_at_2m_speaks_once(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 1000.0  # fixed base time

        # Build 6 ticks: first 3 = persistence warming up, tick 4 = announcement
        frames = [
            make_frame(2.0, [make_det("person")], t + i * 0.1)
            for i in range(6)
        ]

        actions = run_ticks(engine, frames)

        informative_count = actions.count(FusionAction.INFORMATIVE)
        self.assertEqual(informative_count, 1, "Person at 2 m should be announced exactly once")
        # After announcement, should be SILENT for remaining ticks
        first_inf = actions.index(FusionAction.INFORMATIVE)
        post_actions = actions[first_inf + 1:]
        self.assertTrue(
            all(a == FusionAction.SILENT for a in post_actions),
            "After announcement, all subsequent ticks should be SILENT (cooldown)"
        )


class TestScenario2_ChairAtMidRange(unittest.TestCase):
    """Chair at 1.5 m → speak once, silent after."""

    def test_chair_at_1p5m_speaks_once(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 2000.0
        frames = [make_frame(1.5, [make_det("chair")], t + i * 0.1) for i in range(8)]
        actions = run_ticks(engine, frames)
        self.assertEqual(actions.count(FusionAction.INFORMATIVE), 1)


class TestScenario3_ObjectAt50cm(unittest.TestCase):
    """Object at 50 cm → URGENT, no matter what camera sees or doesn't."""

    def test_urgent_no_detections(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.50, [], timestamp=1.0))
        self.assertEqual(result.action, FusionAction.URGENT)

    def test_urgent_with_detections(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.50, [make_det("person")], timestamp=1.0))
        self.assertEqual(result.action, FusionAction.URGENT)

    def test_urgent_at_boundary(self):
        engine = FusionEngine(near_threshold_m=0.60)
        # Exactly at threshold — should NOT be urgent (it's the near boundary)
        result = engine.process(make_frame(0.60, [], timestamp=1.0))
        self.assertNotEqual(result.action, FusionAction.URGENT)
        # Just below — should be urgent
        result2 = engine.process(make_frame(0.599, [], timestamp=1.1))
        self.assertEqual(result2.action, FusionAction.URGENT)


class TestScenario4_UltrasonicCloseNoCamera(unittest.TestCase):
    """Ultrasonic sees close object, camera sees nothing → still URGENT."""

    def test_near_no_vision(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.40, [], timestamp=1.0))
        self.assertEqual(result.action, FusionAction.URGENT)
        self.assertIn("Near zone", result.reason)


class TestScenario5_CameraSeesObjectFar(unittest.TestCase):
    """Camera sees object but ultrasonic reads far → SILENT."""

    def test_far_with_detection_is_silent(self):
        engine = FusionEngine()
        # Even with many ticks of person detection at 3 m, should stay SILENT
        frames = [
            make_frame(3.0, [make_det("person")], timestamp=float(i))
            for i in range(10)
        ]
        actions = run_ticks(engine, frames)
        self.assertTrue(
            all(a == FusionAction.SILENT for a in actions),
            "Detections in far zone must not trigger announcements"
        )


class TestScenario6_FlickeringDetections(unittest.TestCase):
    """Detection flickers frame to frame → suppressed by persistence gate."""

    def test_alternating_detection_suppressed(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 5000.0
        frames = []
        # Alternate: detected, not detected, detected, not detected...
        for i in range(12):
            dets = [make_det("chair")] if i % 2 == 0 else []
            frames.append(make_frame(1.5, dets, t + i * 0.1))

        actions = run_ticks(engine, frames)
        self.assertNotIn(
            FusionAction.INFORMATIVE, actions,
            "Flickering (alternating present/absent) must NOT pass persistence gate"
        )


class TestScenario7_SameObjectTenSeconds(unittest.TestCase):
    """Same object stays in view for 10+ seconds → one announcement only."""

    def test_one_announcement_over_ten_seconds(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 0.0
        tick_interval = 0.1   # 10 Hz
        total_ticks = 110     # 11 seconds

        frames = [
            make_frame(1.5, [make_det("person")], t + i * tick_interval)
            for i in range(total_ticks)
        ]
        actions = run_ticks(engine, frames)

        informative_count = actions.count(FusionAction.INFORMATIVE)
        self.assertEqual(
            informative_count, 1,
            f"Person over 11 s should be announced exactly once (got {informative_count})"
        )


class TestScenario8_TwoObjectsPriority(unittest.TestCase):
    """Two objects detected simultaneously → only higher-priority one announced."""

    def test_person_beats_chair(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 9000.0
        frames = [
            make_frame(1.5, [make_det("chair"), make_det("person")], t + i * 0.1)
            for i in range(6)
        ]
        results = [engine.process(f) for f in frames]
        informative = [r for r in results if r.action == FusionAction.INFORMATIVE]

        # At least one INFORMATIVE must exist
        self.assertGreater(len(informative), 0)

        # The FIRST announcement must be 'person' (highest priority)
        self.assertEqual(
            informative[0].label, "person",
            "Person must be announced before chair"
        )

        # No single tick should announce more than one label
        # (only one INFORMATIVE per tick is possible since process() returns one result)
        # Verify: within cooldown window (12 s), only 1 person announcement
        person_announcements = [r for r in informative if r.label == "person"]
        self.assertEqual(len(person_announcements), 1, "Person should only be announced once within cooldown")


    def test_chair_beats_unknown(self):
        engine = FusionEngine(persistence_ticks=3, cooldown_sec=12.0)
        t = 9500.0
        frames = [
            make_frame(1.5, [make_det("unknown_thing", 0.88), make_det("chair")], t + i * 0.1)
            for i in range(6)
        ]
        results = [engine.process(f) for f in frames]
        informative = [r for r in results if r.action == FusionAction.INFORMATIVE]
        self.assertTrue(len(informative) >= 1)
        self.assertEqual(informative[0].label, "chair")


class TestScenario9_PersonSuddenlyClose(unittest.TestCase):
    """
    Person suddenly appears at near range →
    URGENT must fire on tick 1, NOT waiting for persistence_ticks.
    """

    def test_immediate_urgent_no_persistence_delay(self):
        engine = FusionEngine(persistence_ticks=5, cooldown_sec=12.0)
        # Frame 1: person at 0.45 m, never seen before
        result = engine.process(make_frame(0.45, [make_det("person")], timestamp=1.0))
        self.assertEqual(
            result.action, FusionAction.URGENT,
            "Person at 45 cm must be URGENT on tick 1, before persistence gate fills"
        )

    def test_urgent_fires_even_with_no_camera(self):
        engine = FusionEngine(persistence_ticks=5, cooldown_sec=12.0)
        result = engine.process(make_frame(0.45, [], timestamp=1.0))
        self.assertEqual(result.action, FusionAction.URGENT)


# ---------------------------------------------------------------------------
# State machine: buzzer hysteresis
# ---------------------------------------------------------------------------

class TestStateMachineHysteresis(unittest.TestCase):
    """URGENT buzzer must NOT turn off instantly when distance clears."""

    def setUp(self):
        self.hw = MagicMock(spec=HardwareInterface)
        self.sm = StateMachine(hw=self.hw)

    def test_buzzer_stays_on_during_hysteresis(self):
        from fusion import FusionResult, Zone
        from state_machine import URGENT_HYSTERESIS_SEC

        urgent_result = FusionResult(
            action=FusionAction.URGENT,
            distance_m=0.40,
            zone=Zone.NEAR,
            reason="test"
        )
        silent_result = FusionResult(
            action=FusionAction.SILENT,
            distance_m=2.0,
            zone=Zone.FAR,
            reason="test"
        )

        # Drive into URGENT
        self.sm.update(urgent_result)
        self.assertEqual(self.sm.state, SystemState.URGENT)
        self.hw.buzzer_on.assert_called_once()

        # Immediately switch to SILENT result — should stay URGENT (hysteresis)
        state_after = self.sm.update(silent_result)
        self.assertEqual(
            state_after, SystemState.URGENT,
            "Should stay URGENT during hysteresis window"
        )
        self.hw.buzzer_off.assert_not_called()


class TestStateMachineSingleAnnouncement(unittest.TestCase):
    """INFORMATIVE state: speak() called exactly once per label."""

    def test_speak_called_once(self):
        hw = MagicMock(spec=HardwareInterface)
        sm = StateMachine(hw=hw)

        from fusion import FusionResult, Zone

        inform = FusionResult(
            action=FusionAction.INFORMATIVE,
            label="person",
            zone=Zone.MID,
            reason="test"
        )

        for _ in range(20):
            sm.update(inform)

        hw.speak.assert_called_once_with("person")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
