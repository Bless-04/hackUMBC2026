"""
gemini_narrator.py — Multimodal Scene Description for GuideSense
================================================================
Uses Google Gemini (gemini-3.8-flash) to provide natural, contextual scene descriptions
for visually impaired users wearing the GuideSense chest device.

Key Architecture Principles:
  1. Non-Blocking / Asynchronous:
     Scene understanding runs in a background thread so it NEVER pauses or lags
     the 10 Hz local safety loop (fusion.py + state_machine.py).
  2. Fallback Safe:
     If offline, rate-limited, or if the API key is missing, GuideSense cleanly falls
     back to the local single-word announcement ("person", "chair").
  3. Assistive Prompt Tuning:
     Prompts are specifically engineered for navigation assistance: focus on obstacle
     position, orientation, walking clearance, and immediate environmental context.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

# Primary and fallback Gemini models (Flash-Lite prioritized for high RPM and low latency)
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash", "gemini-flash-latest"]
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

# Debounce interval between Gemini API calls to prevent flooding
MIN_CALL_INTERVAL_SEC = 5.0

# System instruction prompt for assistive vision
SYSTEM_PROMPT = (
    "You are GuideSense, an assistive AI for a blind or low-vision user wearing a chest camera. "
    "The user was alerted to an object in front of them. "
    "Provide a direct, practical, 1-to-2 sentence description focusing on navigation: "
    "where the object is located relative to the user (e.g. directly ahead, on the left/right), "
    "what it is doing or its state, and whether the walking path is clear. "
    "Do not use markdown, bullet points, or emojis. Speak conversationally as voice output."
)


def load_api_key() -> Optional[str]:
    """Load GEMINI_API_KEY from environment variable or .env file."""
    if "GEMINI_API_KEY" in os.environ:
        return os.environ["GEMINI_API_KEY"].strip()

    # Search for .env in current and parent directories
    for path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        return line.split("=", 1)[1].strip("\"' ")
    return None


class GeminiNarrator:
    """
    Asynchronous scene describer powered by Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[list[str]] = None,
    ):
        self.api_key = load_api_key() if api_key is None else api_key
        self.models = models or GEMINI_MODELS
        self._last_call_time = 0.0
        self._lock = threading.Lock()

        if self.api_key:
            print(f"[GeminiNarrator] Initialized with API Key ({self.api_key[:8]}...)")
        else:
            print("[GeminiNarrator] WARNING: No GEMINI_API_KEY found in environment or .env")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def describe_scene(
        self,
        image_bytes: Optional[bytes] = None,
        label: Optional[str] = None,
        distance_m: Optional[float] = None,
    ) -> Optional[str]:
        """
        Synchronous call to Gemini.
        Returns a concise 1-2 sentence description, or None if unavailable.
        """
        if not self.api_key:
            return None

        # Build prompt parts
        parts = []
        context_hint = ""
        if label and distance_m:
            context_hint = f" The system detected a '{label}' at approximately {distance_m:.1f} metres."
        elif label:
            context_hint = f" The system detected a '{label}' ahead."

        prompt_text = f"{SYSTEM_PROMPT}{context_hint}"

        if image_bytes:
            # Multimodal request with image
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_img,
                }
            })
            parts.append({"text": prompt_text})
        else:
            # Text-only contextual prompt
            parts.append({"text": prompt_text + " Give a brief safety guidance statement for this detection."})

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": 300,
                "temperature": 0.3,
            }
        }).encode("utf-8")

        for model in self.models:
            url = GEMINI_API_URL_TEMPLATE.format(model=model, key=self.api_key)
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=6.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    resp_parts = content.get("parts", [])
                    for p in resp_parts:
                        if "text" in p and p["text"].strip():
                            text = p["text"].strip()
                            # Clean up formatting
                            text = text.replace('"', "").replace("\n", " ")
                            return text
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                print(f"[GeminiNarrator] {model} HTTP {e.code}: {err_msg[:120]}")
            except Exception as e:
                print(f"[GeminiNarrator] {model} request failed: {e}")

        return None

    def describe_scene_async(
        self,
        image_bytes: Optional[bytes] = None,
        label: Optional[str] = None,
        distance_m: Optional[float] = None,
        on_complete: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Asynchronous wrapper. Spawns a background daemon thread to query Gemini
        and invokes on_complete(description) when ready.

        Returns True if thread was launched, False if throttled or unavailable.
        """
        now = time.monotonic()
        with self._lock:
            if (now - self._last_call_time) < MIN_CALL_INTERVAL_SEC:
                # Throttled to prevent flooding
                return False
            self._last_call_time = now

        def _worker():
            try:
                description = self.describe_scene(
                    image_bytes=image_bytes,
                    label=label,
                    distance_m=distance_m,
                )
                if description and on_complete:
                    on_complete(description)
            except Exception as e:
                print(f"[GeminiNarrator] Background worker error: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True


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
