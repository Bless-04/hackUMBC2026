# GuideSense — Decision Core

> **Chest-worn navigation aid for low-vision users.**
> This module is the decision-making layer that fuses ultrasonic distance and camera detections into real-time audio/haptic alerts.

---

## What This Module Does

Raw sensor data on its own means nothing — a camera sees a chair, a sonar ping returns 1.5 m. Something has to decide whether the user should hear about it, and when. That's this layer.

Every tick (~10×/sec) it answers one question:

| Situation | Output |
|---|---|
| Nothing relevant nearby | Stay silent |
| New object confirmed at mid-range | Speak its name once, then go quiet |
| Anything within ~60 cm | Buzz continuously until it clears |

---

## File Structure

```
hackUMBC2026/
├── fusion.py          # Core rule engine — distance zones, persistence gate, cooldown
├── state_machine.py   # Hardware output controller — TTS, buzzer, hysteresis
├── main.py            # Integration loop — mock readers + swap-in points for real hardware
└── test_scenarios.py  # 15 tests covering all 9 brief scenarios
```

---

## Quick Start

**Requirements:** Python 3.10+, no external packages needed for mock mode.

```bash
# Run the simulated demo (person walks toward device, then away)
python -X utf8 main.py

# Run all tests
python -X utf8 test_scenarios.py
```

> **Windows note:** Always use the `-X utf8` flag to avoid cp1252 encoding errors in the console output.

The mock demo runs for 10 seconds and prints a live trace:

```
t= 0.00s  dist=4.00m  dets=[—]             fusion=SILENT       state=SILENT
t= 3.01s  dist=3.98m  dets=[person(0.85)]  fusion=SILENT       state=SILENT   ← persistence warming
t= 4.72s  dist=1.59m  dets=[person(0.85)]  fusion=INFORMATIVE  state=INFORMATIVE
[TTS]     'person'
t= 5.82s  dist=0.54m  dets=[person(0.85)]  fusion=URGENT       state=URGENT
[BUZZER]  *** ON ***
t= 7.60s  dist=2.92m  dets=[person(0.85)]  fusion=SILENT       state=SILENT
[BUZZER]  --- off ---
```

---

## How the Logic Works

### 1. Distance Zones (checked first, always)

| Zone | Range | Behaviour |
|---|---|---|
| **NEAR** | < 0.60 m | → **URGENT** immediately, no camera needed |
| **MID** | 0.60 m – 2.00 m | → Announce if object confirmed (see gates below) |
| **FAR** | > 2.00 m | → **SILENT** always |

### 2. The Three Gates (MID zone only)

An object in MID zone must clear **all three** before an announcement fires:

```
[confidence ≥ 50%] → [present for 3 consecutive ticks] → [not announced in last 12 s] → SPEAK
```

- **Confidence gate** — ignore weak detections (< 0.50).
- **Persistence gate** — kills single-frame flicker. One missed tick resets the streak.
- **Cooldown gate** — once announced, that label is silenced for 12 seconds.

### 3. Multi-Object Priority

If multiple objects clear the gates simultaneously, only the **highest-priority** one is announced. Priority order (highest → lowest):

```
person > bicycle/motorcycle > car > dog > chair > table/couch > bed/toilet > anything else
```

### 4. URGENT Fast Path

The NEAR-zone URGENT rule is **completely separate** from the persistence gate. A person stepping in front at 40 cm triggers the buzzer on the very first tick — no waiting for 3-frame confirmation.

### 5. Buzzer Hysteresis

When distance clears back above 0.60 m, the buzzer stays on for **1.5 seconds** before shutting off. This prevents flickering at the boundary if the user is hovering right at the threshold.

---

## Data Contract (Agree with Hardware Teams First)

This module expects two inputs per tick:

```python
# From the CE freshman's ultrasonic serial reader:
distance_m: float          # metres, e.g. 1.47

# From the IT freshman's vision module:
detections: list[Detection]

# Detection fields:
Detection(
    label:      str,                    # e.g. "person", "chair"
    confidence: float,                  # 0.0 – 1.0
    bbox:       tuple[int,int,int,int], # (x1, y1, x2, y2) pixels
    timestamp:  float,                  # time.monotonic()
)
```

---

## Hardware Testing & Subsystem Verification

Before running full end-to-end integration, test each individual hardware subsystem independently:

### 1. Camera Focal Length Calibration (`calibrate.py`)
Calibrate your webcam to compute the exact `FOCAL_LENGTH_PX` constant for pinhole distance calculation:
```bash
# Calibrate using standing person at 2.0 m (default)
python3 calibrate.py --height 170.0 --distance 200.0

# Calibrate using a standard chair (85 cm) at 1.5 m (150 cm)
python3 calibrate.py --height 85.0 --distance 150.0
```
* **Steps**: Press `SPACE` / `ENTER` to freeze the frame, drag a bounding box from top to bottom of the object, and press `ENTER`.
* **Output**: Copy the calculated `FOCAL_LENGTH_PX` value into `distance.py`.

---

### 2. Camera-Only Distance Estimation (`distance.py` / `main.py`)
Test bounding-box distance calculation without physical ultrasonic hardware:
```bash
# Run GuideSense using monocular camera distance estimation
python3 main.py --camera-distance

# Or run with live camera detection and camera distance
python3 main.py --live --duration 20
```

---

### 3. Vision Module Test (`vision.py`)
Test camera capture and real-time object detector:
```bash
python3 vision.py
```
* **Expected Output**: Continuous stream of detections (`label`, `confidence`, `bbox`) at $\ge 10\text{ FPS}$.

---

### 4. Audio / TTS Subsystem Test (`audio.py`)
Test non-blocking speech synthesis:
```bash
python3 audio.py
```
* **Expected Output**: Speaks `"person"`, `"chair"`, `"bicycle"` in order with audible pauses, without hanging the console.

---

### 5. Haptics / Buzzer Test (`haptics.py`)
Test buzzer activation and cleanup:
```bash
python3 haptics.py
```
* **Expected Output**: Buzzer turns ON for 1.0 s, OFF for 1.0 s, and ON for 1.0 s before clean exit.

---

### 6. Ultrasonic Serial Distance Reader (`serial_reader.py`)
Test Arduino/microcontroller USB serial distance streaming:
```bash
python3 serial_reader.py
```
* **Expected Output**: Live stream of distance readings in metres (e.g. `1.45m`, `0.52m`).

---

### 7. End-to-End Live Integration (`main.py`)

Run the full system in your target hardware configuration:

```bash
# 1. Full Real Hardware (Serial distance + Camera Vision + Audio/Haptics)
python3 main.py --real

# 2. Camera-Only Distance Mode (No ultrasonic sensor needed)
python3 main.py --camera-distance

# 3. Partial Real Modes (for incremental testing)
python3 main.py --real-distance   # Real ultrasonic + mock vision
python3 main.py --real-vision     # Real camera vision + mock distance

# 4. Generate Judge CSV Log Evidence
python3 main.py --real --log
```

---

## Swapping in Real Hardware

Everything is modular and swappable in **`main.py`**:

---

## Tunable Parameters

All thresholds live at the top of `fusion.py` and `state_machine.py` — no hunting through logic:

| Parameter | Default | File | Effect |
|---|---|---|---|
| `NEAR_THRESHOLD_M` | `0.60 m` | `fusion.py` | URGENT trigger distance |
| `MID_THRESHOLD_M` | `2.00 m` | `fusion.py` | Max range for announcements |
| `CONFIDENCE_MIN` | `0.50` | `fusion.py` | Minimum detection confidence |
| `PERSISTENCE_TICKS` | `3` | `fusion.py` | Frames needed to confirm object |
| `COOLDOWN_SEC` | `12.0 s` | `fusion.py` | Re-announcement lockout |
| `URGENT_HYSTERESIS_SEC` | `1.5 s` | `state_machine.py` | Buzzer exit delay |
| `TICK_HZ` | `10` | `main.py` | Sensor poll rate |

---

## Test Coverage

```bash
python -X utf8 test_scenarios.py -v
```

All 15 tests pass, covering the 9 required scenarios:

| # | Scenario | Test |
|---|---|---|
| 1 | Person at 2 m → speak once | `test_person_at_2m_speaks_once` |
| 2 | Chair at 1.5 m → speak once, silent after | `test_chair_at_1p5m_speaks_once` |
| 3 | Object at 50 cm → URGENT regardless | `test_urgent_*` (3 subtests) |
| 4 | Close distance, no camera → still URGENT | `test_near_no_vision` |
| 5 | Camera sees something far → stay silent | `test_far_with_detection_is_silent` |
| 6 | Detection flickers frame to frame → suppressed | `test_alternating_detection_suppressed` |
| 7 | Same object stays 10+ seconds → one announcement | `test_one_announcement_over_ten_seconds` |
| 8 | Two objects at once → highest priority only | `test_person_beats_chair`, `test_chair_beats_unknown` |
| 9 | Person suddenly appears close → immediate URGENT | `test_immediate_urgent_no_persistence_delay` |
| + | Buzzer stays on during hysteresis window | `test_buzzer_stays_on_during_hysteresis` |
| + | `speak()` called exactly once per label | `test_speak_called_once` |

---

## Architecture Diagram

```
Ultrasonic Reader ──┐
                    ├──▶  FusionEngine.process(frame)
Camera Detector ────┘          │
                               │ FusionResult(action, label, zone)
                               ▼
                        StateMachine.update(result)
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                  TTS       Buzzer    (Tone)
               (speak once) (continuous) (approaching)
```

---

## Common Issues

**`UnicodeEncodeError: cp1252`** — Run with `python -X utf8 ...` on Windows.

**Object never announced in tests** — Check that frame timestamps don't start at `0.0`. The cooldown gate uses a `None` sentinel for "never announced"; a `0.0` base timestamp is fine as long as frames are not at exactly `t=0`.

**Persistence gate never fills** — If a detection label appears then disappears then reappears (flickering), the streak resets each gap. Three consecutive frames with the same label are required.

**Buzzer won't stop** — The hysteresis window (`URGENT_HYSTERESIS_SEC = 1.5s`) keeps it on briefly after distance clears. This is intentional. Shorten the constant if needed.
