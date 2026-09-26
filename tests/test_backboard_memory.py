"""
tests/test_backboard_memory.py — Unit tests for Backboard.io Memory integration
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from backboard_memory import BackboardMemory


def test_backboard_availability():
    b1 = BackboardMemory(api_key="test-key")
    assert b1.is_cloud_enabled is True

    b2 = BackboardMemory(api_key="")
    assert b2.is_cloud_enabled is False


def test_record_observation_local_fallback():
    mem = BackboardMemory(api_key="")
    mem.record_observation("chair", 1.5, "Wooden chair in hallway")

    assert len(mem._local_history) == 1
    assert mem._local_history[0]["label"] == "chair"
    assert mem._local_history[0]["distance_m"] == 1.5


def test_get_recent_memories_fallback():
    mem = BackboardMemory(api_key="")
    mem.record_observation("chair", 1.5, "Wooden chair")
    mem.record_observation("person", 2.0, "Person walking")

    recent = mem.get_recent_memories(limit=2)
    assert len(recent) == 2
    assert "person" in recent[0]


def test_query_memory_async():
    mem = BackboardMemory(api_key="")
    mem.record_observation("door", 2.0, "Open doorway")

    callback = MagicMock()
    # Mock Gemini so it returns immediately in tests
    with patch("gemini_narrator.GeminiNarrator.describe_scene", return_value="You passed a door 2 meters away."):
        mem.query_memory_async("What is ahead?", on_response=callback)
        time.sleep(0.15)

    callback.assert_called_once()
    assert "door" in callback.call_args[0][0].lower()
