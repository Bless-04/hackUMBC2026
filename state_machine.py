"""
state_machine.py — GuideSense System State Machine
====================================================
Consumes FusionResult objects and manages hardware output:
  • Voice/TTS — INFORMATIVE announcements
  • Urgent tone — continuous during URGENT
  • Hysteresis on URGENT exit so the tone doesn't flicker at the boundary

States:
  SILENT → INFORMATIVE → (APPROACHING →) URGENT
  URGENT → SILENT (with hysteresis)

The APPROACHING state is implemented but skipped if fusion emits URGENT directly.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional, Union

from audio import format_voice_message
from fusion import FusionAction, FusionResult

# ---------------------------------------------------------------------------
# System states
# ---------------------------------------------------------------------------

class SystemState(Enum):
    SILENT      = auto()
    INFORMATIVE = auto()   # speaking an announcement, then cooldown
    APPROACHING = auto()   # optional escalation tone
    URGENT      = auto()   # urgent tone active


# ---------------------------------------------------------------------------
# Output interface — swap implementations in main.py
# ---------------------------------------------------------------------------

class HardwareInterface:
    """
    Thin abstraction over physical outputs.
    Replace each method with audio and alert output in main.py.
    """

    def speak(self, text: str) -> None:
        """Fire TTS or pre-recorded audio."""
        print(f"[SPEAK]   '{text}'")

    def buzzer_on(self) -> None:
        """Start continuous buzzer."""
        print("[BUZZER]  ON")

    def buzzer_off(self) -> None:
        """Stop buzzer."""
        print("[BUZZER]  OFF")

    def tone_approaching(self) -> None:
        """Soft escalating tone for APPROACHING state."""
        print("[TONE]    approaching...")


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

URGENT_HYSTERESIS_SEC  = 1.5   # seconds to stay in URGENT after distance clears
INFORMATIVE_SPEAK_SEC  = 2.0   # how long to keep state=INFORMATIVE before → SILENT
APPROACHING_TONE_HZ    = 3     # tone pulses per second (used by main loop, not here)


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Finite-state machine driven by FusionResult objects.

    Call `update(result)` every tick.
    The machine owns the hardware interface and all timing decisions.
    """

    def __init__(self, hw: Optional[HardwareInterface] = None):
        self.hw: HardwareInterface = hw or HardwareInterface()

        self._state: SystemState = SystemState.SILENT

        # Timestamp when URGENT last fired (for hysteresis)
        self._urgent_last_fired: float = 0.0

        # Timestamp when INFORMATIVE state was entered (for auto-exit)
        self._informative_entered: float = 0.0

        # Label currently being announced (guard against re-entry)
        self._active_label: Optional[str] = None

        # Track buzzer state to avoid redundant on/off calls
        self._buzzer_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SystemState:
        return self._state

    def update(self, result: FusionResult) -> SystemState:
        """
        Drive the state machine with the latest FusionResult.
        Returns the new (or unchanged) SystemState.
        """
        now = time.monotonic()

        # ----------------------------------------------------------------
        # URGENT path — highest priority, processed first
        # ----------------------------------------------------------------
        if result.action == FusionAction.URGENT:
            self._enter_urgent(now)
            return self._state

        # ----------------------------------------------------------------
        # Exiting URGENT — apply hysteresis so buzzer doesn't flicker
        # ----------------------------------------------------------------
        if self._state == SystemState.URGENT:
            elapsed_since_urgent = now - self._urgent_last_fired
            if elapsed_since_urgent < URGENT_HYSTERESIS_SEC:
                # Stay in URGENT for hysteresis window even if fusion says otherwise
                self._ensure_buzzer_on()
                return self._state
            else:
                # Hysteresis window passed — safe to exit URGENT
                self._exit_urgent()

        # ----------------------------------------------------------------
        # INFORMATIVE path
        # ----------------------------------------------------------------
        if result.action == FusionAction.INFORMATIVE and result.label:
            if self._state != SystemState.INFORMATIVE or self._active_label != result.label:
                self._enter_informative(result, now)
            return self._state

        # ----------------------------------------------------------------
        # Auto-exit INFORMATIVE after speak window
        # ----------------------------------------------------------------
        if self._state == SystemState.INFORMATIVE:
            if (now - self._informative_entered) >= INFORMATIVE_SPEAK_SEC:
                self._enter_silent()
            return self._state

        # ----------------------------------------------------------------
        # SILENT (default)
        # ----------------------------------------------------------------
        if self._state != SystemState.SILENT:
            self._enter_silent()

        return self._state

    def reset(self) -> None:
        """Hard-reset to SILENT (use between test scenarios)."""
        self._exit_urgent()
        self._state = SystemState.SILENT
        self._urgent_last_fired = 0.0
        self._informative_entered = 0.0
        self._active_label = None

    # ------------------------------------------------------------------
    # State transition helpers
    # ------------------------------------------------------------------

    def _enter_urgent(self, now: float) -> None:
        if self._state != SystemState.URGENT:
            print(f"[STATE]   {self._state.name} -> URGENT")
        self._state = SystemState.URGENT
        self._urgent_last_fired = now
        self._ensure_buzzer_on()

    def _exit_urgent(self) -> None:
        self._ensure_buzzer_off()

    def _enter_informative(self, result_or_label: Union[FusionResult, str], now: float) -> None:
        if isinstance(result_or_label, str):
            label = result_or_label
            speech_msg = label
        else:
            label = result_or_label.label or ""
            speech_msg = format_voice_message(result=result_or_label)

        print(f"[STATE]   {self._state.name} -> INFORMATIVE ('{label}')")
        self._state = SystemState.INFORMATIVE
        self._active_label = label
        self._informative_entered = now
        self.hw.speak(speech_msg)

    def _enter_silent(self) -> None:
        if self._state != SystemState.SILENT:
            print(f"[STATE]   {self._state.name} -> SILENT")
        self._state = SystemState.SILENT
        self._active_label = None
        self._ensure_buzzer_off()

    # ------------------------------------------------------------------
    # Buzzer helpers — idempotent
    # ------------------------------------------------------------------

    def _ensure_buzzer_on(self) -> None:
        if not self._buzzer_active:
            self.hw.buzzer_on()
            self._buzzer_active = True

    def _ensure_buzzer_off(self) -> None:
        if self._buzzer_active:
            self.hw.buzzer_off()
            self._buzzer_active = False
