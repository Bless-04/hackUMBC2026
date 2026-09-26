"""
tests/test_fusion.py — FusionEngine unit tests
================================================
Covers all nine brief scenarios plus zone boundary and confidence gate.
Each test is tagged with @pytest.mark markers for targeted runs:

    pytest -m scenario          # all nine brief scenarios
    pytest -m "scenario and not slow"
    pytest -m regression        # guard tests added after specific bug fixes
    pytest -m fusion            # everything in this file
"""
from __future__ import annotations

import pytest

from fusion import FusionAction, FusionEngine

# Fixtures and helpers are injected from tests/conftest.py
from tests.conftest import make_det, make_frame, run_ticks

# ===========================================================================
# Scenario 1 — Person at 2 m → speak once, then silent
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario1_PersonAtMidRange:
    """Person at 2 m — announced exactly once, cooldown blocks re-announcement."""

    def test_person_at_2m_announces_once(self, engine):
        t = 1000.0
        frames = [make_frame(2.0, [make_det("person")], t + i * 0.1) for i in range(6)]
        actions = run_ticks(engine, frames)

        assert actions.count(FusionAction.INFORMATIVE) == 1, (
            "Person at 2 m should be announced exactly once"
        )

    def test_cooldown_silences_after_announcement(self, engine):
        t = 1000.0
        frames = [make_frame(2.0, [make_det("person")], t + i * 0.1) for i in range(6)]
        actions = run_ticks(engine, frames)

        first_inf = actions.index(FusionAction.INFORMATIVE)
        post = actions[first_inf + 1:]
        assert all(a == FusionAction.SILENT for a in post), (
            "All ticks after announcement must be SILENT (cooldown active)"
        )


# ===========================================================================
# Scenario 2 — Chair at 1.5 m → speak once, silent after
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario2_ChairAtMidRange:

    def test_chair_at_1p5m_announces_once(self, engine):
        t = 2000.0
        frames = [make_frame(1.5, [make_det("chair")], t + i * 0.1) for i in range(8)]
        actions = run_ticks(engine, frames)
        assert actions.count(FusionAction.INFORMATIVE) == 1


# ===========================================================================
# Scenario 3 — Object at 50 cm → URGENT regardless of camera
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario3_ObjectAt50cm:

    def test_urgent_with_no_detections(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.50, [], timestamp=1.0))
        assert result.action == FusionAction.URGENT

    def test_urgent_with_detections(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.50, [make_det("person")], timestamp=1.0))
        assert result.action == FusionAction.URGENT

    def test_at_exact_threshold_is_not_urgent(self):
        """Exactly at 0.60 m is MID (inclusive boundary), not NEAR."""
        engine = FusionEngine(near_threshold_m=0.60)
        result = engine.process(make_frame(0.60, [], timestamp=1.0))
        assert result.action != FusionAction.URGENT

    def test_just_below_threshold_is_urgent(self):
        engine = FusionEngine(near_threshold_m=0.60)
        result = engine.process(make_frame(0.599, [], timestamp=1.0))
        assert result.action == FusionAction.URGENT


# ===========================================================================
# Scenario 4 — Ultrasonic close, camera sees nothing → still URGENT
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario4_UltrasonicCloseNoCamera:

    def test_near_no_vision_is_urgent(self):
        engine = FusionEngine()
        result = engine.process(make_frame(0.40, [], timestamp=1.0))
        assert result.action == FusionAction.URGENT
        assert "Near zone" in result.reason


# ===========================================================================
# Scenario 5 — Camera sees object but ultrasonic reads far → SILENT
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario5_CameraSeesObjectFarAway:

    def test_far_zone_always_silent(self):
        engine = FusionEngine()
        frames = [
            make_frame(3.0, [make_det("person")], timestamp=float(i))
            for i in range(10)
        ]
        actions = run_ticks(engine, frames)
        assert all(a == FusionAction.SILENT for a in actions), (
            "Detections in far zone (>2m) must never trigger announcements"
        )


# ===========================================================================
# Scenario 6 — Detection flickers frame to frame → suppressed
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario6_FlickeringDetections:

    def test_alternating_presence_suppressed(self, engine):
        """Chair appears every other tick — persistence gate must block announcement."""
        t = 5000.0
        frames = [
            make_frame(1.5, [make_det("chair")] if i % 2 == 0 else [], t + i * 0.1)
            for i in range(12)
        ]
        actions = run_ticks(engine, frames)
        assert FusionAction.INFORMATIVE not in actions, (
            "Alternating present/absent detections must not pass persistence gate"
        )

    def test_two_consecutive_not_enough(self, engine):
        """Only 2 consecutive ticks (< persistence_ticks=3) must not announce."""
        t = 6000.0
        # 2 ticks detected, then gap
        frames = [
            make_frame(1.5, [make_det("chair")], t + 0.0),
            make_frame(1.5, [make_det("chair")], t + 0.1),
            make_frame(1.5, [],                  t + 0.2),  # gap resets streak
        ]
        actions = run_ticks(engine, frames)
        assert FusionAction.INFORMATIVE not in actions


# ===========================================================================
# Scenario 7 — Same object 10+ seconds → exactly one announcement
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario7_SameObjectTenSeconds:

    def test_one_announcement_only(self, engine):
        """The headline demo scenario: person in view for 11 s → announced once."""
        frames = [
            make_frame(1.5, [make_det("person")], i * 0.1)
            for i in range(110)   # 11 seconds at 10 Hz
        ]
        actions = run_ticks(engine, frames)
        count = actions.count(FusionAction.INFORMATIVE)
        assert count == 1, f"Expected 1 announcement, got {count}"


# ===========================================================================
# Scenario 8 — Two objects at once → highest priority only
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario8_MultiObjectPriority:

    def test_person_announced_before_chair(self, engine):
        """When person + chair both visible, first announcement must be 'person'."""
        t = 9000.0
        frames = [
            make_frame(1.5, [make_det("chair"), make_det("person")], t + i * 0.1)
            for i in range(6)
        ]
        results = [engine.process(f) for f in frames]
        informative = [r for r in results if r.action == FusionAction.INFORMATIVE]

        assert len(informative) >= 1
        assert informative[0].label == "person", (
            f"First announcement must be 'person', got '{informative[0].label}'"
        )

    def test_person_not_announced_twice_in_cooldown(self, engine):
        t = 9000.0
        frames = [
            make_frame(1.5, [make_det("chair"), make_det("person")], t + i * 0.1)
            for i in range(6)
        ]
        results = [engine.process(f) for f in frames]
        person_count = sum(
            1 for r in results
            if r.action == FusionAction.INFORMATIVE and r.label == "person"
        )
        assert person_count == 1, "Person should only be announced once within cooldown"

    def test_chair_beats_unknown_object(self, engine):
        t = 9500.0
        frames = [
            make_frame(1.5, [make_det("unknown_thing", 0.88), make_det("chair")], t + i * 0.1)
            for i in range(6)
        ]
        results = [engine.process(f) for f in frames]
        informative = [r for r in results if r.action == FusionAction.INFORMATIVE]

        assert len(informative) >= 1
        assert informative[0].label == "chair", (
            "Chair (known priority) must beat unlisted object"
        )


# ===========================================================================
# Scenario 9 — Person suddenly appears close → immediate URGENT, no delay
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.scenario
class TestScenario9_PersonSuddenlyClose:

    def test_urgent_fires_on_first_tick(self, strict_engine):
        """persistence_ticks=5 but URGENT must fire on tick 1 — fast path is separate."""
        result = strict_engine.process(make_frame(0.45, [make_det("person")], timestamp=1.0))
        assert result.action == FusionAction.URGENT, (
            "NEAR zone must trigger URGENT on tick 1, not wait for persistence gate"
        )

    def test_urgent_fires_without_camera(self, strict_engine):
        result = strict_engine.process(make_frame(0.45, [], timestamp=1.0))
        assert result.action == FusionAction.URGENT


# ===========================================================================
# Zone boundary regression tests  (added after off-by-one bug)
# ===========================================================================

@pytest.mark.fusion
@pytest.mark.regression
class TestZoneBoundaries:
    """Guard against the ≤ vs < zone boundary bug that caused Scenario 1 to fail."""

    @pytest.mark.parametrize("distance_m,expected_zone_name", [
        (0.59,  "NEAR"),
        (0.60,  "MID"),    # at boundary — must be MID not NEAR
        (0.61,  "MID"),
        (1.00,  "MID"),
        (2.00,  "MID"),    # at upper boundary — must be MID not FAR
        (2.01,  "FAR"),
        (3.00,  "FAR"),
    ])
    def test_zone_at_distance(self, distance_m, expected_zone_name):
        engine = FusionEngine()
        zone = engine._compute_zone(distance_m)
        assert zone.name == expected_zone_name, (
            f"dist={distance_m}m → expected {expected_zone_name}, got {zone.name}"
        )


# ===========================================================================
# Confidence gate
# ===========================================================================

@pytest.mark.fusion
class TestConfidenceGate:

    def test_low_confidence_suppressed(self, engine):
        """Detections below confidence_min must not count toward persistence."""
        t = 100.0
        frames = [
            make_frame(1.5, [make_det("person", confidence=0.30)], t + i * 0.1)
            for i in range(10)
        ]
        actions = run_ticks(engine, frames)
        assert FusionAction.INFORMATIVE not in actions

    def test_exactly_at_threshold_passes(self, engine):
        """Confidence exactly at CONFIDENCE_MIN (0.50) must pass."""
        t = 200.0
        frames = [
            make_frame(1.5, [make_det("person", confidence=0.50)], t + i * 0.1)
            for i in range(6)
        ]
        actions = run_ticks(engine, frames)
        assert FusionAction.INFORMATIVE in actions

    def test_below_threshold_by_one_thousandth_fails(self, engine):
        t = 300.0
        frames = [
            make_frame(1.5, [make_det("person", confidence=0.499)], t + i * 0.1)
            for i in range(10)
        ]
        actions = run_ticks(engine, frames)
        assert FusionAction.INFORMATIVE not in actions


# ===========================================================================
# Spatial Directional Awareness Tests
# ===========================================================================

@pytest.mark.fusion
class TestSpatialDirectionalAwareness:
    """Spatial direction calculation across zones, boundaries, and resolutions."""

    def test_object_clearly_on_left(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (10, 50, 100, 200)  # x_center = 55.0, 55/640 = 0.086 < 0.33
        direction = compute_direction(bbox, frame_width=640)
        assert direction == Direction.LEFT

        det = Detection(label="person", confidence=0.85, bbox=bbox, frame_width=640)
        assert det.direction == Direction.LEFT

    def test_object_in_center(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (260, 50, 380, 200)  # x_center = 320.0, 320/640 = 0.50 (0.33 - 0.66)
        direction = compute_direction(bbox, frame_width=640)
        assert direction == Direction.CENTER

        det = Detection(label="chair", confidence=0.80, bbox=bbox, frame_width=640)
        assert det.direction == Direction.CENTER

    def test_object_clearly_on_right(self):
        from fusion import Detection, Direction, compute_direction

        bbox = (500, 50, 620, 200)  # x_center = 560.0, 560/640 = 0.875 > 0.66
        direction = compute_direction(bbox, frame_width=640)
        assert direction == Direction.RIGHT

        det = Detection(label="vehicle", confidence=0.90, bbox=bbox, frame_width=640)
        assert det.direction == Direction.RIGHT

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
            left_box = (0, 0, int(width * 0.20), 100)
            assert compute_direction(left_box, frame_width=width) == Direction.LEFT

            center_box = (int(width * 0.40), 0, int(width * 0.60), 100)
            assert compute_direction(center_box, frame_width=width) == Direction.CENTER

            right_box = (int(width * 0.80), 0, width, 100)
            assert compute_direction(right_box, frame_width=width) == Direction.RIGHT

    def test_zone_boundaries(self):
        from fusion import Direction, compute_direction

        width = 1000

        # Boundary at 0.33
        assert compute_direction((328, 0, 328, 100), frame_width=width) == Direction.LEFT
        assert compute_direction((330, 0, 330, 100), frame_width=width) == Direction.CENTER
        assert compute_direction((332, 0, 332, 100), frame_width=width) == Direction.CENTER

        # Boundary at 0.66
        assert compute_direction((658, 0, 658, 100), frame_width=width) == Direction.CENTER
        assert compute_direction((660, 0, 660, 100), frame_width=width) == Direction.CENTER
        assert compute_direction((662, 0, 662, 100), frame_width=width) == Direction.RIGHT

    def test_fusion_result_preserves_direction_and_confidence(self):
        from fusion import Detection, Direction, FusionAction, FusionEngine, SensorFrame, Zone

        engine = FusionEngine(persistence_ticks=1, cooldown_sec=12.0)
        det_left = Detection(label="person", confidence=0.82, bbox=(10, 50, 100, 200), frame_width=640)
        frame = SensorFrame(distance_m=1.2, detections=[det_left], timestamp=100.0, frame_width=640)

        result = engine.process(frame)

        assert result.action == FusionAction.INFORMATIVE
        assert result.label == "person"
        assert result.distance_m == 1.2
        assert result.zone == Zone.MID
        assert result.direction == Direction.LEFT
        assert result.confidence == 0.82


# ===========================================================================
# Directional Voice Output Formatting Tests
# ===========================================================================

@pytest.mark.fusion
class TestDirectionalVoiceOutput:
    """Natural-language voice message formatting with spatial awareness."""

    def test_informative_voice_messages(self):
        from audio import format_voice_message
        from fusion import Direction, FusionAction

        msg_left = format_voice_message(label="person", direction=Direction.LEFT, action=FusionAction.INFORMATIVE)
        assert msg_left == "Person on your left."

        msg_center = format_voice_message(label="chair", direction=Direction.CENTER, action=FusionAction.INFORMATIVE)
        assert msg_center == "Chair in front of you."

        msg_right = format_voice_message(label="vehicle", direction=Direction.RIGHT, action=FusionAction.INFORMATIVE)
        assert msg_right == "Vehicle on your right."

    def test_urgent_hazard_voice_messages(self):
        from audio import format_voice_message
        from fusion import Direction, FusionAction

        msg_urgent_left = format_voice_message(label="person", direction=Direction.LEFT, action=FusionAction.URGENT)
        assert msg_urgent_left == "Stop. Person on your left."

        msg_urgent_obstacle = format_voice_message(label=None, action=FusionAction.URGENT)
        assert msg_urgent_obstacle == "Stop. Obstacle ahead."

    def test_state_machine_speaks_directional_message(self, mock_hw, sm):
        from fusion import Direction, FusionAction, FusionResult, Zone

        result = FusionResult(
            action=FusionAction.INFORMATIVE,
            label="person",
            distance_m=1.2,
            zone=Zone.MID,
            direction=Direction.LEFT,
            confidence=0.82,
        )

        sm.update(result)
        mock_hw.speak.assert_called_once_with("Person on your left.")


# ===========================================================================
# GuideSense HUD Tests
# ===========================================================================

@pytest.mark.fusion
class TestGuideSenseHUD:
    """Verifies live visual HUD overlay generation and state rendering."""

    def test_hud_draws_on_different_resolutions(self):
        import numpy as np

        from fusion import Detection, Direction, FusionAction, FusionResult, SensorFrame, Zone
        from hud import GuideSenseHUD
        from state_machine import SystemState

        hud = GuideSenseHUD(features={
            "Camera": "Real",
            "Arduino": "Connected",
            "Gemini": "Active",
            "Backboard": "Active",
            "Logging": "Active",
        })

        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
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

            rendered = hud.draw_hud(canvas, frame, result, SystemState.INFORMATIVE)
            assert rendered.shape == (h, w, 3)
            assert np.any(rendered > 0)

    def test_hud_handles_all_system_states(self):
        import numpy as np

        from fusion import Detection, FusionAction, FusionResult, SensorFrame, Zone
        from hud import GuideSenseHUD
        from state_machine import SystemState

        hud = GuideSenseHUD()
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = SensorFrame(distance_m=0.4, detections=[Detection("chair", 0.85, (50, 100, 200, 400))], frame_width=640)

        # 1. URGENT state
        res_urgent = FusionResult(action=FusionAction.URGENT, label="chair", distance_m=0.4, zone=Zone.NEAR)
        out_urgent = hud.draw_hud(canvas, frame, res_urgent, SystemState.URGENT)
        assert out_urgent.shape == (480, 640, 3)

        # 2. SILENT state
        res_silent = FusionResult(action=FusionAction.SILENT, distance_m=3.5, zone=Zone.FAR)
        out_silent = hud.draw_hud(canvas, frame, res_silent, SystemState.SILENT)
        assert out_silent.shape == (480, 640, 3)

