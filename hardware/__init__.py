"""
hardware — GuideSense Audio & Haptic Feedback Adapters
======================================================
Speech synthesis (local pyttsx3 & cloud ElevenLabs) and dynamic haptic buzzer tones.
"""

from hardware.audio import AudioOutput, format_voice_message
from hardware.audio_playback import play_wav_bytes
from hardware.eleven_audio import ElevenLabsVoice, pcm_to_wav_bytes
from hardware.haptics import HapticOutput

__all__ = [
    "AudioOutput",
    "ElevenLabsVoice",
    "HapticOutput",
    "format_voice_message",
    "pcm_to_wav_bytes",
    "play_wav_bytes",
]
