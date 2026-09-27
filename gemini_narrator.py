"""
gemini_narrator.py — Backward-compatibility alias for services.gemini_narrator
=============================================================================
This file re-exports all symbols from `services.gemini_narrator` to maintain a
single canonical source of truth for Gemini narration and prompts.
"""

from __future__ import annotations

from services.gemini_narrator import (
    GEMINI_API_URL_TEMPLATE,
    GEMINI_MODELS,
    MIN_CALL_INTERVAL_SEC,
    SYSTEM_PROMPT,
    GeminiNarrator,
    load_api_key,
)

__all__ = [
    "GEMINI_API_URL_TEMPLATE",
    "GEMINI_MODELS",
    "MIN_CALL_INTERVAL_SEC",
    "SYSTEM_PROMPT",
    "GeminiNarrator",
    "load_api_key",
]

# ---------------------------------------------------------------------------
# Standalone CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    narrator = GeminiNarrator()
    if not narrator.is_available:
        print("Please set GEMINI_API_KEY in .env before running.")
    else:
        print("Testing text contextual description for 'chair' at 1.5m...")
        result = narrator.describe_scene(label="chair", distance_m=1.5)
        print(f"Result:\n-> \"{result}\"")
