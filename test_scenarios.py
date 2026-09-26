"""
test_scenarios.py — GuideSense Nine-Scenario Validation Suite
==============================================================
Run with:  python test_scenarios.py
All nine scenarios from the brief are exercised here.
No external dependencies beyond the standard library.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

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
# Spatial Directional Awareness Tests
# ---------------------------------------------------------------------------

class TestSpatialDirectionalAwareness(unittest.TestCase):
    """Spatial direction calculation across zones, boundaries, and resolutions."""

    def test_object_clearly_on_left(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (10, 50, 100, 200)  # x_center = 55.0, 55/640 = 0.086 < 0.33
        direction = compute_direction(bbox, frame_width=640)
        self.assertEqual(direction, Direction.LEFT)

        det = Detection(label="person", confidence=0.85, bbox=bbox, frame_width=640)
        self.assertEqual(det.direction, Direction.LEFT)

    def test_object_in_center(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (260, 50, 380, 200)  # x_center = 320.0, 320/640 = 0.50 (0.33 - 0.66)
        direction = compute_direction(bbox, frame_width=640)
        self.assertEqual(direction, Direction.CENTER)

        det = Detection(label="chair", confidence=0.80, bbox=bbox, frame_width=640)
        self.assertEqual(det.direction, Direction.CENTER)

    def test_object_clearly_on_right(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (500, 50, 620, 200)  # x_center = 560.0, 560/640 = 0.875 > 0.66
        direction = compute_direction(bbox, frame_width=640)
        self.assertEqual(direction, Direction.RIGHT)

        det = Detection(label="vehicle", confidence=0.90, bbox=bbox, frame_width=640)
        self.assertEqual(det.direction, Direction.RIGHT)

    def test_different_camera_resolutions(self):
        from fusion import Direction, compute_direction

        resolutions = [
            (300, 300),    # MobileNet native
            (640, 480),    # Standard VGA
            (1280, 720),   # 720p HD
            (1920, 1080),  # 1080p Full HD
            (3840, 2160),  # 4K UHD
        ]

        for width, _ in resolutions:
            # 10% from left edge -> LEFT
            left_box = (0, 0, int(width * 0.20), 100)
            self.assertEqual(
                compute_direction(left_box, frame_width=width),
                Direction.LEFT,
                f"Failed for width={width} left",
            )

            # 50% center -> CENTER
            center_box = (int(width * 0.40), 0, int(width * 0.60), 100)
            self.assertEqual(
                compute_direction(center_box, frame_width=width),
                Direction.CENTER,
                f"Failed for width={width} center",
            )

            # 90% right edge -> RIGHT
            right_box = (int(width * 0.80), 0, width, 100)
            self.assertEqual(
                compute_direction(right_box, frame_width=width),
                Direction.RIGHT,
                f"Failed for width={width} right",
            )

    def test_zone_boundaries(self):
        from fusion import Direction, compute_direction

        width = 1000  # for clean normalized arithmetic

        # Left/Center boundary at 0.33
        self.assertEqual(compute_direction((328, 0, 328, 100), frame_width=width), Direction.LEFT)    # 0.328 < 0.33
        self.assertEqual(compute_direction((330, 0, 330, 100), frame_width=width), Direction.CENTER)  # 0.330 == 0.33
        self.assertEqual(compute_direction((332, 0, 332, 100), frame_width=width), Direction.CENTER)  # 0.332 > 0.33

        # Center/Right boundary at 0.66
        self.assertEqual(compute_direction((658, 0, 658, 100), frame_width=width), Direction.CENTER)  # 0.658 < 0.66
        self.assertEqual(compute_direction((660, 0, 660, 100), frame_width=width), Direction.CENTER)  # 0.660 == 0.66
        self.assertEqual(compute_direction((662, 0, 662, 100), frame_width=width), Direction.RIGHT)   # 0.662 > 0.66

    def test_fusion_result_preserves_direction_and_confidence(self):
        from fusion import Detection, Direction, FusionAction, FusionEngine, SensorFrame, Zone

        engine = FusionEngine(persistence_ticks=1, cooldown_sec=12.0)
        det_left = Detection(label="person", confidence=0.82, bbox=(10, 50, 100, 200), frame_width=640)
        frame = SensorFrame(distance_m=1.2, detections=[det_left], timestamp=100.0, frame_width=640)

        result = engine.process(frame)

        self.assertEqual(result.action, FusionAction.INFORMATIVE)
        self.assertEqual(result.label, "person")
        self.assertEqual(result.distance_m, 1.2)
        self.assertEqual(result.zone, Zone.MID)
        self.assertEqual(result.direction, Direction.LEFT)
        self.assertEqual(result.confidence, 0.82)


# ---------------------------------------------------------------------------
# Directional Voice Output Formatting Tests
# ---------------------------------------------------------------------------

class TestDirectionalVoiceOutput(unittest.TestCase):
    """Natural-language voice message formatting with spatial awareness."""

    def test_informative_voice_messages(self):
        from audio import format_voice_message
        from fusion import Direction, FusionAction

        # Person on left
        msg_left = format_voice_message(label="person", direction=Direction.LEFT, action=FusionAction.INFORMATIVE)
        self.assertEqual(msg_left, "Person on your left.")

        # Chair in center
        msg_center = format_voice_message(label="chair", direction=Direction.CENTER, action=FusionAction.INFORMATIVE)
        self.assertEqual(msg_center, "Chair in front of you.")

        # Vehicle on right
        msg_right = format_voice_message(label="vehicle", direction=Direction.RIGHT, action=FusionAction.INFORMATIVE)
        self.assertEqual(msg_right, "Vehicle on your right.")

    def test_urgent_hazard_voice_messages(self):
        from audio import format_voice_message
        from fusion import Direction, FusionAction

        # Stop. Person on your left.
        msg_urgent_left = format_voice_message(label="person", direction=Direction.LEFT, action=FusionAction.URGENT)
        self.assertEqual(msg_urgent_left, "Stop. Person on your left.")

        # Stop. Obstacle ahead. (sonar only, no vision label)
        msg_urgent_obstacle = format_voice_message(label=None, action=FusionAction.URGENT)
        self.assertEqual(msg_urgent_obstacle, "Stop. Obstacle ahead.")

    def test_state_machine_speaks_directional_message(self):
        from fusion import Direction, FusionAction, FusionResult, Zone

        hw = MagicMock(spec=HardwareInterface)
        sm = StateMachine(hw=hw)

        result = FusionResult(
            action=FusionAction.INFORMATIVE,
            label="person",
            distance_m=1.2,
            zone=Zone.MID,
            direction=Direction.LEFT,
            confidence=0.82,
        )

        sm.update(result)
        hw.speak.assert_called_once_with("Person on your left.")


class TestGuideSenseHUD(unittest.TestCase):
    """Verifies HUD drawing across resolutions, states, distance zones, and directions."""

    def setUp(self):
        import numpy as np

        from hud import GuideSenseHUD
        self.np = np
        self.hud = GuideSenseHUD(features={
            "Camera": "Real",
            "Arduino": "Connected",
            "Gemini": "Active",
            "Backboard": "Active",
            "Logging": "Active",
        })

    def test_hud_draws_on_different_resolutions(self):
        from fusion import Detection, Direction, FusionAction, FusionResult, SensorFrame, Zone
        from state_machine import SystemState

        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            canvas = self.np.zeros((h, w, 3), dtype=self.np.uint8)
            det = Detection("person", 0.92, (int(w * 0.1), int(h * 0.2), int(w * 0.3), int(h * 0.8)), frame_width=w)
            frame = SensorFrame(distance_m=1.5, detections=[det], frame_width=w)
            result = FusionResult(
                action=FusionAction.INFORMATIVE,
                label="person",
                distance_m=1.5,
                zone=Zone.MID,
                direction=Direction.LEFT,
                confidence=0.92,
            )

            rendered = self.hud.draw_hud(canvas, frame, result, SystemState.INFORMATIVE)
            self.assertEqual(rendered.shape, (h, w, 3))
            # Image should have drawn elements (not all zeros)
            self.assertTrue(self.np.any(rendered > 0))

    def test_hud_handles_all_system_states(self):
        from fusion import Detection, FusionAction, FusionResult, SensorFrame, Zone
        from state_machine import SystemState

        canvas = self.np.zeros((480, 640, 3), dtype=self.np.uint8)
        frame = SensorFrame(distance_m=0.4, detections=[Detection("chair", 0.85, (50, 100, 200, 400))], frame_width=640)

        # 1. URGENT state
        res_urgent = FusionResult(action=FusionAction.URGENT, label="chair", distance_m=0.4, zone=Zone.NEAR)
        out_urgent = self.hud.draw_hud(canvas, frame, res_urgent, SystemState.URGENT)
        self.assertEqual(out_urgent.shape, (480, 640, 3))

        # 2. SILENT state
        res_silent = FusionResult(action=FusionAction.SILENT, distance_m=3.5, zone=Zone.FAR)
        out_silent = self.hud.draw_hud(canvas, frame, res_silent, SystemState.SILENT)
        self.assertEqual(out_silent.shape, (480, 640, 3))

    def test_hud_toggle_and_paused_overlay(self):
        from fusion import FusionAction, FusionResult, SensorFrame, Zone
        from state_machine import SystemState

        canvas = self.np.zeros((480, 640, 3), dtype=self.np.uint8)
        frame = SensorFrame(distance_m=1.8, detections=[], frame_width=640)
        result = FusionResult(action=FusionAction.SILENT, distance_m=1.8, zone=Zone.MID)

        # HUD disabled (clean view)
        self.hud.show_hud = False
        out_clean = self.hud.draw_hud(canvas, frame, result, SystemState.SILENT)
        self.assertEqual(out_clean.shape, (480, 640, 3))

        # HUD paused
        self.hud.show_hud = True
        self.hud.is_paused = True
        out_paused = self.hud.draw_hud(canvas, frame, result, SystemState.SILENT)
        self.assertTrue(self.np.any(out_paused > 0))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
