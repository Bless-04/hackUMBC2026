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
