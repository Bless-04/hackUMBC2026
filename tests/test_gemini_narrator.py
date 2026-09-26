"""
tests/test_gemini_narrator.py — Tests for Gemini Multimodal Scene Narrator
==========================================================================
Uses mocks so CI never makes real network calls or consumes API quota.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from gemini_narrator import GeminiNarrator


def test_availability():
    narrator_with_key = GeminiNarrator(api_key="fake-test-key")
    assert narrator_with_key.is_available is True

    narrator_no_key = GeminiNarrator(api_key="")
    assert narrator_no_key.is_available is False


@patch("urllib.request.urlopen")
def test_describe_scene_mocked(mock_urlopen):
    # Mock successful Gemini API JSON response
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "A person is standing 1.5 meters ahead on your left."}]
                }
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    narrator = GeminiNarrator(api_key="fake-test-key")
    result = narrator.describe_scene(label="person", distance_m=1.5)

    assert result == "A person is standing 1.5 meters ahead on your left."
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_describe_scene_async_callback(mock_urlopen):
    import time

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "There is a chair directly ahead."}]
                }
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    narrator = GeminiNarrator(api_key="fake-test-key")
    callback = MagicMock()

    launched = narrator.describe_scene_async(label="chair", distance_m=1.8, on_complete=callback)
    assert launched is True

    # Give background thread a moment to finish
    time.sleep(0.1)
    callback.assert_called_once_with("There is a chair directly ahead.")


def test_throttling_prevents_spam():
    narrator = GeminiNarrator(api_key="fake-test-key")
    # First call launches
    launched1 = narrator.describe_scene_async(label="person", distance_m=1.5)
    assert launched1 is True

    # Immediate second call should be throttled
    launched2 = narrator.describe_scene_async(label="person", distance_m=1.4)
    assert launched2 is False
