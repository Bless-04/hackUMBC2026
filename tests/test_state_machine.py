"""
tests/test_state_machine.py — StateMachine unit tests
=======================================================
Tests hardware output, state transitions, buzzer hysteresis,
and the single-announcement guarantee.

Markers:
    @pytest.mark.state_machine  — all tests in this file
    @pytest.mark.regression     — bug-guard tests
"""
from __future__ import annotations

import time

import pytest

from fusion import FusionAction, FusionResult, Zone
from state_machine import URGENT_HYSTERESIS_SEC, SystemState

# ===========================================================================
# URGENT → buzzer on
# ===========================================================================

@pytest.mark.state_machine
class TestUrgentState:

    def test_urgent_turns_buzzer_on(self, sm, mock_hw, urgent_result):
        sm.update(urgent_result)
        mock_hw.buzzer_on.assert_called_once()

    def test_state_is_urgent_after_urgent_result(self, sm, urgent_result):
        sm.update(urgent_result)
        assert sm.state == SystemState.URGENT

    def test_buzzer_called_only_once_on_repeat_urgent(self, sm, mock_hw, urgent_result):
        """Idempotent — repeated URGENT results must not call buzzer_on twice."""
        sm.update(urgent_result)
        sm.update(urgent_result)
        sm.update(urgent_result)
        mock_hw.buzzer_on.assert_called_once()


# ===========================================================================
# Hysteresis — buzzer stays on briefly after distance clears
# ===========================================================================

@pytest.mark.state_machine
@pytest.mark.regression
class TestUrgentHysteresis:
    """
    Regression guard for the hysteresis window on URGENT exit.
    The buzzer must NOT turn off the moment fusion returns SILENT —
    it should stay on for URGENT_HYSTERESIS_SEC seconds.
    """

    def test_buzzer_stays_on_immediately_after_distance_clears(
        self, sm, mock_hw, urgent_result, silent_result
    ):
        sm.update(urgent_result)                     # enter URGENT
        state_after = sm.update(silent_result)       # distance clears
        assert state_after == SystemState.URGENT, (
            "Must stay URGENT during hysteresis window"
        )
        mock_hw.buzzer_off.assert_not_called()

    def test_buzzer_off_after_hysteresis_expires(
        self, sm, mock_hw, urgent_result, silent_result, monkeypatch
    ):
        """After the hysteresis window passes, the next SILENT tick exits URGENT."""
        sm.update(urgent_result)

        # Monkey-patch monotonic so the state machine thinks time has passed
        past = time.monotonic() - URGENT_HYSTERESIS_SEC - 0.1
        monkeypatch.setattr(sm, "_urgent_last_fired", past)

        state_after = sm.update(silent_result)
        assert state_after == SystemState.SILENT
        mock_hw.buzzer_off.assert_called_once()


# ===========================================================================
# INFORMATIVE — speak once, then go silent
# ===========================================================================

@pytest.mark.state_machine
class TestInformativeState:

    def test_speak_called_once(self, sm, mock_hw, informative_result):
        for _ in range(20):
            sm.update(informative_result)
        mock_hw.speak.assert_called_once_with("person")

    def test_state_is_informative_after_announce(self, sm, informative_result):
        sm.update(informative_result)
        assert sm.state == SystemState.INFORMATIVE

    def test_different_label_triggers_new_speak(self, sm, mock_hw):
        """After cooldown, a different label should fire a new TTS call."""
        result_a = FusionResult(
            action=FusionAction.INFORMATIVE, label="chair", zone=Zone.MID, reason=""
        )
        result_b = FusionResult(
            action=FusionAction.INFORMATIVE, label="person", zone=Zone.MID, reason=""
        )
        sm.update(result_a)
        sm.update(result_b)
        assert mock_hw.speak.call_count == 2

    def test_auto_exit_to_silent_after_speak_window(
        self, sm, informative_result, silent_result, monkeypatch
    ):
        """After INFORMATIVE_SPEAK_SEC, the next SILENT tick transitions back to SILENT."""
        from state_machine import INFORMATIVE_SPEAK_SEC
        sm.update(informative_result)
        assert sm.state == SystemState.INFORMATIVE

        # Wind the clock forward past the speak window
        past = time.monotonic() - INFORMATIVE_SPEAK_SEC - 0.1
        monkeypatch.setattr(sm, "_informative_entered", past)

        # A SILENT result (not INFORMATIVE) triggers the auto-exit check
        sm.update(silent_result)
        assert sm.state == SystemState.SILENT


# ===========================================================================
# SILENT state
# ===========================================================================

@pytest.mark.state_machine
class TestSilentState:

    def test_starts_silent(self, sm):
        assert sm.state == SystemState.SILENT

    def test_silent_result_stays_silent(self, sm, silent_result):
        sm.update(silent_result)
        assert sm.state == SystemState.SILENT

    def test_no_hardware_calls_when_always_silent(self, sm, mock_hw, silent_result):
        for _ in range(10):
            sm.update(silent_result)
        mock_hw.speak.assert_not_called()
        mock_hw.buzzer_on.assert_not_called()
        mock_hw.buzzer_off.assert_not_called()


# ===========================================================================
# Reset
# ===========================================================================

@pytest.mark.state_machine
class TestReset:

    def test_reset_turns_buzzer_off(self, sm, mock_hw, urgent_result):
        sm.update(urgent_result)
        sm.reset()
        mock_hw.buzzer_off.assert_called()

    def test_reset_returns_to_silent(self, sm, urgent_result):
        sm.update(urgent_result)
        sm.reset()
        assert sm.state == SystemState.SILENT
