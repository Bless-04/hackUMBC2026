
## GuideSense — CS Senior Brief: Fusion Engine, State Machine, Integration

**Your role:** You own the decision-making core of the system and final integration of all three tracks. The freshmen are building well-specified hardware/vision modules; you're building the one piece that requires real judgment calls, and you're responsible for wiring everyone's work together correctly.

**What the product is:** A chest-worn device for low-vision users. An ultrasonic sensor gives distance, a camera + pretrained object detector gives object identity. Your job is the layer that decides, every tick, whether the system should stay silent, speak once calmly, or alert urgently. This decision layer — not the sensors — is the actual engineering contribution and the whole pitch.

**Your two files: `fusion.py` and `state_machine.py`, plus final `main.py` integration.**

### Fusion logic (`fusion.py`) — the core rule, in order:
1. **Distance zone always checked first:** distance < ~60cm → near (safety-critical). 60-200cm → mid. Above → far.
2. **Near zone always wins, unconditionally** — if distance is near, the answer is URGENT regardless of what the camera sees or doesn't see. Never gate this behind vision confirmation.
3. **In mid zone:** if a detected object clears a confidence threshold AND has been consistently detected for a few consecutive ticks (persistence gate — this kills single-frame flicker), AND it hasn't been announced recently (cooldown) → speak it once.
4. **If multiple objects detected at once:** pick only the highest-importance one (person > chair > other) — never announce a list.
5. **Never let persistence-gating delay the near-distance URGENT path** — that's a separate, faster-firing rule specifically so a suddenly-appearing close object (like a person stepping in front) isn't missed while waiting for confirmation.

### State machine (`state_machine.py`):
Four states: **SILENT → INFORMATIVE → APPROACHING → URGENT**
- **SILENT:** default, nothing happening.
- **INFORMATIVE:** new relevant object confirmed at mid-range — speak once, then cooldown timer starts (~10-15 sec) blocking re-announcement of that same object.
- **APPROACHING** (optional if time is short): distance closing but not yet urgent — a soft escalating tone, skip this state entirely if you're behind schedule.
- **URGENT:** distance breached near-threshold — buzzer fires continuously until distance clears back out, with a short hysteresis window on exit so it doesn't flicker in and out right at the boundary.

**The single most important behavior to get right:** if the same object stays in view for 10+ seconds, the system says its name **once**, not repeatedly. This is your strongest demo moment and your strongest data point for judges — test it explicitly.

### Integration responsibility:
- **Data contract** (agree on this with both freshmen in the first 15 minutes): a `Detection` object (class name, confidence, bounding box, timestamp) comes from the IT freshman's vision module; a distance value + timestamp comes from the CE freshman's serial reader. Your fusion function consumes both.
- **Start immediately against mock data** — don't wait on real hardware. Fake a distance stream (a number that decreases then increases, simulating someone walking toward then away from something) and a fake detection list. Build and test your logic against that first; swap in real inputs later without changing your core logic.
- **When real modules are ready**, you plug them in and resolve any mismatches between what you expected and what they actually built.

### Test it against these nine scenarios before calling it done:
Person at 2m → speak once. Chair at 1.5m → speak once, silent after. Object at 50cm → urgent, no matter what. Ultrasonic sees something close but camera sees nothing → still urgent (distance always wins). Camera sees something but ultrasonic reads far → stay silent, don't announce distant objects. Detection flickers frame to frame → suppressed by persistence gate. Same object stays 10+ seconds → one announcement only. Two objects at once → announce only the higher-priority one. Person suddenly appears close → immediate urgent, don't let persistence-gating delay it.

**Bottom line for him:** if the fusion/state-machine logic works — silent when nothing matters, calm once when something new appears, unmistakably urgent when something's genuinely close — the team wins on technical depth regardless of how polished anything else is. This is where to spend the best thinking, not on model selection or wiring.