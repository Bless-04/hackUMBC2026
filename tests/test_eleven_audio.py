"""
tests/test_eleven_audio.py — Unit tests for ElevenLabs Voice integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hardware.eleven_audio import ElevenLabsVoice, pcm_to_wav_bytes


def test_eleven_voice_availability():
    v1 = ElevenLabsVoice(api_key="test-key")
    assert v1.is_available is True

    v2 = ElevenLabsVoice(api_key="")
    assert v2.is_available is False


def test_pcm_to_wav_bytes_creates_valid_wav_header():
    # 100 samples of 16-bit silence (200 bytes)
    dummy_pcm = b"\x00\x00" * 100
    wav_bytes = pcm_to_wav_bytes(dummy_pcm, sample_rate=24000)

    # Standard RIFF WAV starts with "RIFF" and contains "WAVE"
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert len(wav_bytes) > len(dummy_pcm)


@patch("urllib.request.urlopen")
def test_generate_wav_mocked(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"\x00\x00" * 50
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    voice = ElevenLabsVoice(api_key="fake-test-key")
    wav = voice.generate_wav("Test speech")

    assert wav is not None
    assert wav.startswith(b"RIFF")


@patch.object(ElevenLabsVoice, "speak", return_value=True)
def test_speak_async_launches_thread(mock_speak):
    import time
    voice = ElevenLabsVoice(api_key="fake-test-key")
    callback = MagicMock()

    launched = voice.speak_async("Hello async", on_done=callback)
    assert launched is True

    time.sleep(0.1)
    mock_speak.assert_called_once_with("Hello async")
    callback.assert_called_once()
