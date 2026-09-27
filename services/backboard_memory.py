"""
backboard_memory.py — Persistent Spatial & Session Memory via Backboard.io
==========================================================================
Integrates Backboard.io to give GuideSense durable long-term spatial memory:
  - Stores persistent landmarks, obstacle events, and spatial navigation context.
  - Uses Backboard SDK (or REST) for durable Memory & RAG storage.
  - Automatically queries memory and passes context to Gemini + ElevenLabs.
  - Operates asynchronously: NEVER blocks the 10 Hz local safety loop.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Callable, Optional


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
        assistant_name: str = "GuideSense Assistant",
    ):
        self.api_key = load_backboard_key() if api_key is None else api_key
        self.assistant_name = assistant_name
        self.assistant_id = None
        self._local_history: list[dict] = []
        self._lock = threading.Lock()
        self._client = None

        if self.api_key:
            print(f"[BackboardMemory] Initialized with key ({self.api_key[:8]}...)")
            self._init_client_async()
        else:
            print("[BackboardMemory] Local session memory mode (set BACKBOARD_API_KEY in .env for cloud persistence)")

    @property
    def is_cloud_enabled(self) -> bool:
        return bool(self.api_key)

    def _init_client_async(self) -> None:
        """Initialize BackboardClient in background to get/create assistant ID."""
        def _setup():
            try:
                from backboard import BackboardClient
                client = BackboardClient(api_key=self.api_key)

                async def _get_assistant():
                    assistants = await client.list_assistants()
                    for a in assistants:
                        if getattr(a, "name", "") == self.assistant_name:
                            await client.aclose()
                            return a.assistant_id
                    # Create if not found
                    new_a = await client.create_assistant(
                        name=self.assistant_name,
                        description="Spatial memory assistant for GuideSense low-vision navigation",
                        system_prompt="Track landmarks, obstacles, and navigation history for a visually impaired user.",
                    )
                    await client.aclose()
                    return new_a.assistant_id

                self.assistant_id = asyncio.run(_get_assistant())
                print(f"[BackboardMemory] Connected to Assistant ID: {self.assistant_id}")
            except Exception as e:
                print(f"[BackboardMemory] Cloud initialization note: {e}")

        t = threading.Thread(target=_setup, daemon=True)
        t.start()

    def record_observation(
        self,
        label: str,
        distance_m: float,
        description: Optional[str] = None,
    ) -> None:
        """
        Record an observed object/hazard into memory asynchronously.
        """
        entry = {
            "timestamp": time.time(),
            "label": label,
            "distance_m": round(distance_m, 2),
            "description": description or f"Detected {label} at {distance_m:.1f}m",
        }
        with self._lock:
            self._local_history.append(entry)

        # Sync to Backboard cloud asynchronously
        if self.is_cloud_enabled:
            def _sync():
                # Wait briefly for assistant_id if still initializing
                for _ in range(10):
                    if self.assistant_id and self._client:
                        break
                    time.sleep(0.3)

                if self.assistant_id:
                    try:
                        from backboard import BackboardClient

                        async def _add():
                            client = BackboardClient(api_key=self.api_key)
                            content = (
                                f"Navigation observation: A {label} was detected at approximately "
                                f"{distance_m:.1f} metres ahead. Context: {entry['description']}"
                            )
                            await client.add_memory(
                                assistant_id=self.assistant_id,
                                content=content,
                            )
                            await client.aclose()

                        asyncio.run(_add())
                    except Exception as e:
                        print(f"[BackboardMemory] Cloud memory save error: {e}")

            t = threading.Thread(target=_sync, daemon=True)
            t.start()

    def get_recent_memories(self, limit: int = 5) -> list[str]:
        """Fetch the most recent memory statements."""
        if self.assistant_id:
            try:
                from backboard import BackboardClient

                async def _fetch():
                    client = BackboardClient(api_key=self.api_key)
                    res = await client.get_memories(assistant_id=self.assistant_id)
                    await client.aclose()
                    return [m.content for m in getattr(res, "memories", [])][:limit]

                return asyncio.run(_fetch())
            except Exception:
                pass

        # Fallback to local history
        with self._lock:
            return [
                f"{h['label']} at {h['distance_m']}m"
                for h in reversed(self._local_history[-limit:])
            ]

    def query_memory_async(self, question: str, on_response: Callable[[str], None]) -> None:
        """
        Asynchronously answers user questions using Backboard persistent memory
        combined with Gemini for natural phrasing.
        """
        def _worker():
            # 1. If Backboard cloud is connected, try Backboard's semantic search + Gemini completion
            if self.is_cloud_enabled and self.assistant_id:
                try:
                    from backboard import BackboardClient

                    async def _query_backboard():
                        client = BackboardClient(api_key=self.api_key)
                        # Semantic search for relevant memories
                        res = await client.search_memories(
                            assistant_id=self.assistant_id, query=question
                        )
                        raw_mems = res.get("memories", []) if isinstance(res, dict) else getattr(res, "memories", [])
                        mems = [m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") for m in raw_mems]

                        # Fallback to general recent memories if search was empty
                        if not mems:
                            mres = await client.get_memories(assistant_id=self.assistant_id)
                            mems = [m.content for m in getattr(mres, "memories", [])][:5]

                        if not mems:
                            await client.aclose()
                            return None

                        # Ask assistant thread with Gemini 3.1 Flash Lite (high RPM, low latency)
                        thread = await client.create_thread(assistant_id=self.assistant_id)
                        ctx = " ".join(mems)
                        prompt = (
                            f"Context from navigation memory: {ctx}\n"
                            f"User question: {question}\n"
                            "Answer in 1 concise sentence for a blind user navigating with GuideSense."
                        )
                        resp = None
                        for model_name in ["gemini-3.1-flash-lite", "gemini-3.8-flash"]:
                            try:
                                resp = await client.send_message(
                                    thread_id=thread.thread_id,
                                    content=prompt,
                                    llm_provider="google",
                                    model_name=model_name,
                                )
                                if resp and getattr(resp, "content", None):
                                    break
                            except Exception:
                                continue

                        await client.aclose()
                        return resp.content if resp else None

                    reply = asyncio.run(_query_backboard())
                    if reply:
                        on_response(reply)
                        return
                except Exception as e:
                    print(f"[BackboardMemory] Cloud query note: {e}")

            # 2. Fallback to local history + direct Gemini narrator
            memories = self.get_recent_memories(limit=5)
            if not memories:
                on_response("I have not recorded any navigation landmarks in memory yet.")
                return

            mem_context = "; ".join(memories)

            # Try generating a conversational reply via Gemini if available
            try:
                from services.gemini_narrator import GeminiNarrator
                narrator = GeminiNarrator()
                if narrator.is_available:
                    prompt = (
                        f"The user is asking: '{question}'. "
                        f"Here is their recent navigation memory history: {mem_context}. "
                        f"Answer their question in 1 concise sentence based on these memories."
                    )
                    reply = narrator.describe_scene(label=prompt)
                    if reply:
                        on_response(reply)
                        return
            except Exception:
                pass

            # Fallback direct reply
            on_response(f"Based on your recent journey: {mem_context}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Standalone CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mem = BackboardMemory()
    print("Waiting 1.5s for Backboard cloud sync...")
    time.sleep(1.5)

    print("Recording observation of 'chair' at 1.5m...")
    mem.record_observation("chair", 1.5, "A wooden chair in the hallway path")
    time.sleep(1.0)

    done_event = threading.Event()

    def print_reply(text):
        print(f"\nAI Memory Query Response:\n-> \"{text}\"")
        done_event.set()

    print("Querying memory: 'Did I see any chairs?'...")
    mem.query_memory_async("Did I see any chairs?", on_response=print_reply)
    done_event.wait(timeout=6.0)
    print("Done.")
