"""
backboard_memory.py — Persistent Spatial & Session Memory via Backboard.io
==========================================================================
Integrates Backboard.io to give GuideSense durable long-term memory:
  - Tracks navigation history, obstacles seen, landmarks passed, and user habits.
  - Allows contextual memory recall across threads (e.g. "Where did I leave my chair?").
  - Works with Backboard REST API or SDK, with graceful local fallback if offline.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

BACKBOARD_API_URL = "https://api.backboard.io/v1/messages"


def load_backboard_key() -> Optional[str]:
    """Load BACKBOARD_API_KEY from environment or .env file."""
    for env_var in ["BACKBOARD_API_KEY", "BACKBOARD_KEY"]:
        if env_var in os.environ and os.environ[env_var].strip():
            return os.environ[env_var].strip()

    for path in [".env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("BACKBOARD_API_KEY=") or line.startswith("BACKBOARD_KEY="):
                        return line.split("=", 1)[1].strip("\"' ")
    return None


class BackboardMemory:
    """
    Persistent memory manager for GuideSense using Backboard.io.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        self.api_key = load_backboard_key() if api_key is None else api_key
        self.assistant_id = assistant_id or os.environ.get("BACKBOARD_ASSISTANT_ID", "guidesense-assistant")
        self.thread_id = thread_id or os.environ.get("BACKBOARD_THREAD_ID")
        self._local_history: list[dict] = []
        self._lock = threading.Lock()

        if self.api_key:
            print(f"[BackboardMemory] Initialized with key ({self.api_key[:8]}...) assistant={self.assistant_id}")
        else:
            print("[BackboardMemory] Local session memory mode (set BACKBOARD_API_KEY in .env for cloud persistence)")

    @property
    def is_cloud_enabled(self) -> bool:
        return bool(self.api_key)

    def record_observation(
        self,
        label: str,
        distance_m: float,
        description: Optional[str] = None,
    ) -> None:
        """
        Record an observed object/hazard into memory.
        """
        entry = {
            "timestamp": time.time(),
            "label": label,
            "distance_m": round(distance_m, 2),
            "description": description or f"Detected {label} at {distance_m:.1f}m",
        }
        with self._lock:
            self._local_history.append(entry)

        # Sync to Backboard cloud asynchronously if key is configured
        if self.is_cloud_enabled:
            def _sync():
                prompt = (
                    f"Observation update: The user's chest camera detected a '{label}' "
                    f"at {distance_m:.1f} metres ahead. Context: {entry['description']}"
                )
                self.send_memory_update(prompt)

            t = threading.Thread(target=_sync, daemon=True)
            t.start()

    def send_memory_update(self, content: str) -> Optional[str]:
        """Sends an observation or message to Backboard.io."""
        if not self.api_key:
            return None

        payload = {
            "content": content,
            "assistant_id": self.assistant_id,
            "memory": "Auto",
        }
        if self.thread_id:
            payload["thread_id"] = self.thread_id

        req = urllib.request.Request(
            BACKBOARD_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "thread_id" in data:
                    self.thread_id = data["thread_id"]
                return data.get("content")
        except Exception as e:
            print(f"[BackboardMemory] Cloud sync error: {e}")
            return None

    def query_memory_async(self, question: str, on_response: Callable[[str], None]) -> None:
        """
        Asynchronously answers user questions using local or cloud Backboard memory.
        """
        def _worker():
            if self.is_cloud_enabled:
                reply = self.send_memory_update(question)
                if reply:
                    on_response(reply)
                    return

            # Local fallback memory search
            with self._lock:
                if not self._local_history:
                    on_response("I have not detected any objects in this session yet.")
                    return
                latest = self._local_history[-1]
                on_response(f"The last object detected was a {latest['label']} about {latest['distance_m']} metres away.")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Standalone CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mem = BackboardMemory()
    mem.record_observation("chair", 1.5, "A wooden chair in the hallway")
    print(f"Recorded observation. Total in memory: {len(mem._local_history)}")

    def print_reply(text):
        print(f"Memory Query Response: \"{text}\"")

    mem.query_memory_async("What did I see recently?", on_response=print_reply)
    time.sleep(0.5)
