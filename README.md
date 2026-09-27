# GuideSense — Decision Core & Developer Manual

> **Chest-worn assistive navigation device for low-vision users.**  
> Fuses computer vision and proximity data to deliver real-time audio and haptic guidance: staying silent when clear, speaking calm contextual announcements once, and alerting urgently when hazards are near.

---

## Browser Dashboard

Run `python -m ui` from the repository root, then open **http://127.0.0.1:8765**.
The new dashboard lives in [`ui/`](ui/README.md) and includes a hardware-free demo,
live camera detections, distance/alert status, optional AI/voice controls, and
exportable session activity. Start in demo mode, or choose **Configure → Live camera**
to test your Logitech webcam. The existing CLI and `vision.py` are unchanged.

See the [dashboard setup and hardware success checklist](ui/README.md) for details.

## Table of Contents
1. [System Architecture & Hardware Setup](#1-system-architecture--hardware-setup)
2. [Team Ownership Matrix](#2-team-ownership-matrix)
3. [Developer Setup & Environment](#3-developer-setup--environment)
4. [Step-by-Step Developer Implementation Guides](#4-step-by-step-developer-implementation-guides)
   - [CE Track: Arduino & Breadboard Controller](#ce-track-arduino--breadboard-controller)
   - [IT Track: Vision, Audio & Haptics](#it-track-vision-audio--haptics)
   - [CS Senior Track: Fusion, State Machine & Integration](#cs-senior-track-fusion-state-machine--integration)
5. [Data Contracts & Interface Signatures](#5-data-contracts--interface-signatures)
6. [Core Decision Logic & Priority Rules](#6-core-decision-logic--priority-rules)
7. [AI Cloud Suite: Gemini + ElevenLabs + Backboard](#7-ai-cloud-suite-gemini--elevenlabs--backboard)
8. [Operational Modes & CLI Reference](#8-operational-modes--cli-reference)
9. [Testing & Quality Assurance Guide](#9-testing--quality-assurance-guide)
10. [Linting, Code Quality & CI](#10-linting-code-quality--ci)
11. [Troubleshooting & Gotchas](#11-troubleshooting--gotchas)

---

## 1. System Architecture & Hardware Setup

### Physical Hardware Inventory
- **Compute:** Raspberry Pi (executes main sensing loop, fusion engine, state machine, and computer vision)
- **Vision & Depth:** Logitech Webcam (USB)
- **Audio Output:** JBL Speaker (connected via 3.5mm AUX, USB, or Bluetooth)
- **Hardware Indicators:** Arduino Uno/Nano + Breadboard + Status LEDs (Green, Yellow, Red) + optional Potentiometer
- **Wearable:** Chest harness mounting the Pi, camera, breadboard, and speaker

### The Ultrasonic Pivot (Monocular Bounding-Box Distance Estimation)
Because a dedicated physical ultrasonic sensor was unavailable, the Logitech webcam provides **both** object recognition and metric distance estimation:
1. `vision.py` detects objects and returns bounding boxes `(x1, y1, x2, y2)`.
2. `distance_estimator.py` calculates object proximity using the vertical height ratio of the bounding box relative to the camera frame:
   - **Height > 65% of frame:** Object is dangerously close $\rightarrow$ **$< 0.60\,\text{m}$ (NEAR / URGENT)**
   - **Height 20% to 65% of frame:** Object is in walking path $\rightarrow$ **$0.60\,\text{m} - 2.00\,\text{m}$ (MID / INFORMATIVE)**
   - **Height < 20% of frame:** Object is far away $\rightarrow$ **$> 2.00\,\text{m}$ (FAR / SILENT)**
3. If an Arduino with a potentiometer is wired on the breadboard, it can also broadcast `D:<cm>\n` over USB serial to provide manual distance override for live demonstrations.

```
                  ┌────────────────────────────────────────────────────────┐
                  │                    LOGITECH WEBCAM                     │
                  └───────────┬────────────────────────────────┬───────────┘
                              │                                │
                     Raw Video Frames                  Bounding Box Height
                              │                                │
                              ▼                                ▼
                  ┌───────────────────────┐        ┌───────────────────────┐
                  │       vision.py       │        │ distance_estimator.py │
                  │  (MobileNet-SSD VOC)  │        │ (or serial_reader.py) │
                  └───────────┬───────────┘        └───────────┬───────────┘
                              │                                │
                     list[Detection]                      distance_m
                              │                                │
                              └───────────────┬────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │        fusion.py        │
                                 │   (Zone & Gate Rules)   │
                                 └────────────┬────────────┘
                                              │ FusionResult
                                              ▼
                                 ┌─────────────────────────┐
                                 │    state_machine.py     │
                                 │ (Hysteresis & Outputs)  │
                                 └────────────┬────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                     ▼                     ▼
             ┌─────────────────────┐┌───────────────────┐┌─────────────────────┐
             │      audio.py       ││    haptics.py     ││  serial_reader.py   │
             │     JBL Speaker     ││    JBL Speaker    ││ Arduino Breadboard  │
             │ (TTS Announcements) ││ (880Hz Siren Tone)││   (Status LEDs)     │
             └─────────────────────┘└───────────────────┘└─────────────────────┘
```

---

## 2. Team Ownership Matrix

| Role | Teammate | Files Owned | Responsibilities |
|---|---|---|---|
| **CS Senior** | Integration Lead | `fusion.py`, `state_machine.py`, `distance_estimator.py`, `main.py`, `tests/` | Decision core logic, priority ranking, persistence & cooldown gates, state transitions, hardware abstraction, CI/CD, mock integration. |
| **CE Freshman** | Hardware & Sensors | `arduino_guidesense/arduino_guidesense.ino`, `serial_reader.py` | Flash Arduino sketch, wire breadboard status LEDs (Green/Yellow/Red) and optional potentiometer dial, physical chest harness assembly. |
| **IT Freshman** | Vision, Audio, Logging | `vision.py`, `audio.py`, `haptics.py`, `logger.py`, `eleven_audio.py` | Logitech webcam frame capture & MobileNet-SSD detection, pyttsx3/ElevenLabs voice on JBL speaker, urgent audio alert loop on JBL speaker, CSV session logging. |

---

## 3. Developer Setup & Environment

GuideSense uses standard Python `pip` and `venv` (with optional `uv` support).

### 1. Clone & Set Up Virtual Environment (Standard `pip` — Recommended)
```bash
git clone https://github.com/your-org/hackUMBC2026.git
cd hackUMBC2026

# Create virtual environment:
python -m venv .venv

# Activate virtual environment:
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / Raspberry Pi / macOS:
source .venv/bin/activate

# Install all development and testing dependencies:
pip install -r requirements.txt
```

*(Optional for `uv` users: `uv sync --dev` also works).*

### 3. Environment Variables (`.env`)
Copy the example template to create your `.env` file:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
```ini
# Google Gemini API Key (for multimodal contextual scene description)
GEMINI_API_KEY=your_gemini_api_key_here

# ElevenLabs API Key (for studio-quality voice output through JBL speaker)
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here

# Backboard.io API Key (optional — for persistent spatial memory)
BACKBOARD_API_KEY=your_backboard_api_key_here
```
*(Note: `.env` is ignored in `.gitignore` on line 154 to protect secrets).*

### 4. Linux / Raspberry Pi System Permissions
If running on the Raspberry Pi:
```bash
# Allow serial access to Arduino without sudo:
sudo usermod -a -G dialout $USER

# Install ALSA audio tools and espeak for local TTS fallback:
sudo apt update && sudo apt install -y alsa-utils espeak
```

---

## 4. Step-by-Step Developer Implementation Guides

### CE Track: Arduino & Breadboard Controller

#### Step 1: Flash Arduino Sketch
Open `arduino_guidesense/arduino_guidesense.ino` in the Arduino IDE and upload it to the board.

#### Step 2: Breadboard Wiring
| Arduino Pin | Breadboard Component | Function |
|---|---|---|
| **Pin 2** | Green LED (with 220Ω resistor to GND) | Lights when state is `SILENT` (path clear) |
| **Pin 3** | Yellow LED (with 220Ω resistor to GND) | Lights when state is `INFORMATIVE` (mid-range object) |
| **Pin 4** | Red LED (with 220Ω resistor to GND) | Lights when state is `URGENT` (hazard detected) |
| **Pin A0** | Potentiometer center wiper pin | Optional: manual distance dial (sends `D:<cm>\n` over serial) |
| **5V / GND** | Power rails | Connect Arduino 5V and GND to breadboard rails |

#### Step 3: Implement & Test `serial_reader.py`
In `serial_reader.py`:
1. Verify `SERIAL_PORT`: `"COM3"` / `"COM4"` on Windows, `"/dev/ttyUSB0"` or `"/dev/ttyACM0"` on Raspberry Pi.
2. Uncomment the `pyserial` reading logic in `SerialDistanceReader.read()`.
3. Test standalone:
```bash
python -X utf8 serial_reader.py
```
*Expected Output:* Prints live distance readings in metres (`0.450 m`, `1.520 m`) and responds to serial state commands.

---

### IT Track: Vision, Audio & Haptics

#### Step 1: Camera & Model Setup (`vision.py`)
1. Connect the Logitech webcam via USB.
2. Download MobileNet-SSD Caffe weights into the project root:
   - `MobileNetSSD_deploy.prototxt`
   - `MobileNetSSD_deploy.caffemodel`
3. Install the project dependencies: `pip install -r requirements.txt`.
4. Confirm both model files are present before starting the camera reader.
5. Test standalone:
```bash
python -X utf8 vision.py
```
*Expected Output:* Prints detected objects (`person`, `chair`), confidence, and bounding box coordinates at $\ge 10\text{ FPS}$.

#### Step 2: Audio & Voice Setup (`audio.py` / `eleven_audio.py`)
- GuideSense automatically routes speech to **ElevenLabs** if `ELEVEN_LABS_API_KEY` is present in `.env`, falling back to local `pyttsx3` offline.
- Test standalone:
```bash
python -X utf8 audio.py
```
*Expected Output:* Natural voice speaks `"GuideSense initialized with Gemini, ElevenLabs, and Backboard."` through the JBL speaker.

#### Step 3: Urgent Alert Setup (`haptics.py`)
- Emits a pulsing 880 Hz urgency tone via a non-blocking background thread when `URGENT` triggers.
- Test standalone:
```bash
python -X utf8 haptics.py
```
*Expected Output:* Emits urgency tone for 1 second, pauses for 1 second, and repeats.

#### Raspberry Pi Camera + AI Hardware Test (No Arduino Required)

This test uses only the **Raspberry Pi**, **Logitech USB camera**, and **JBL speaker**. The Arduino, breadboards, and jumper wires are not required because object distance is estimated from the camera bounding boxes.

The active test pipeline is:

```text
Logitech camera -> MobileNet-SSD -> camera distance estimate -> fusion/state logic
                -> Gemini guidance -> ElevenLabs/JBL audio -> Backboard memory
```

##### 1. Prepare the Raspberry Pi

Connect the Logitech camera, pair or cable the JBL speaker, activate the Python virtual environment, and install the dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
sudo apt update
sudo apt install -y alsa-utils espeak
```

Confirm that Linux can see the camera and an audio output device:

```bash
ls /dev/video*
aplay -l
```

**Success:** At least one camera device such as `/dev/video0` is listed, and the JBL speaker or its active audio interface appears in the playback-device list.

Place both MobileNet-SSD files in the repository root:

```text
MobileNetSSD_deploy.prototxt
MobileNetSSD_deploy.caffemodel
```

Create `.env` from `.env.example` and set the services you want to exercise:

```ini
GEMINI_API_KEY=your_real_key
ELEVEN_LABS_API_KEY=your_real_key
BACKBOARD_API_KEY=your_real_key
```

Gemini, ElevenLabs, and cloud Backboard testing requires an internet connection. Backboard can still retain session-local observations without its cloud key, and audio falls back to local `pyttsx3` if ElevenLabs is unavailable.

##### 2. Verify Camera Detection and Speed

```bash
python -X utf8 vision.py
```

Stand in front of the camera and place supported objects such as a chair in view. Press `Ctrl+C` to stop.

**Success:**

- The terminal prints labels such as `person` or `chair`, confidence scores, and pixel bounding boxes.
- The confidence is at least `0.40` when a result is returned.
- The reported rate is at least `10 FPS` on the Raspberry Pi.
- The program exits cleanly with `Ctrl+C` and releases the camera.

##### 3. Verify Camera Distance and Decision Logic

This stage deliberately leaves real audio and cloud services off so the local safety logic can be observed first:

```bash
python -X utf8 main.py --camera-distance --duration 30
```

Walk slowly toward the camera while remaining fully visible.

**Success:**

- Startup reports `REAL VisionReader` and `CAMERA BOUNDING-BOX DISTANCE ESTIMATOR`.
- Each terminal row shows `dist`, detected objects, `fusion`, and `state`.
- A stable mid-range object must be detected for three consecutive ticks before the state becomes `INFORMATIVE`.
- Moving very close so the bounding box occupies more than approximately 65% of the frame height produces a distance below `0.60m` and triggers `URGENT` immediately.
- Moving away returns the system to `SILENT` after the urgent-state hysteresis expires.
- Rows are appended to `guidesense_log.csv` for later review.

##### 4. Verify the Full Camera + AI + JBL Path

The `--real` flag enables the real audio and urgent-tone outputs. When combined with `--camera-distance`, the Arduino distance reader is not used.

```bash
python -X utf8 main.py \
  --camera-distance \
  --gemini \
  --backboard \
  --real \
  --duration 60
```

**Successful startup includes messages indicating:**

- `REAL VisionReader`
- `CAMERA BOUNDING-BOX DISTANCE ESTIMATOR`
- `REAL AudioOutput + HapticOutput`
- `Google Gemini Multimodal Scene Narrator ACTIVE`
- Backboard persistent navigation memory is active, or that local-session fallback is being used

**Successful behavior:**

- A confirmed mid-range detection becomes `INFORMATIVE`, is recorded by Backboard, and produces spoken guidance through the JBL speaker.
- With a valid ElevenLabs key, startup reports the ElevenLabs voice backend; without it, local `pyttsx3` speaks instead.
- A near object becomes `URGENT` on the first near-distance tick and starts the non-blocking 880 Hz speaker alarm.
- Camera detection and the 10 Hz safety loop continue while cloud narration and speech run in background threads.

For an unlimited test, replace `--duration 60` with `--forever` and stop it with `Ctrl+C`.

> **Multimodal Gemini Vision:** `main.py` automatically captures the live camera frame, encodes it to JPEG bytes, and passes it into `GeminiNarrator.describe_scene_async(image_bytes=...)` alongside the estimated distance and MobileNet label. This enables true multimodal visual scene understanding (orientation, state, context) while keeping local 10 Hz obstacle safety unblocked.

##### Common Failure Indicators

| Symptom | Meaning / next check |
|---|---|
| `VisionReader failed` followed by mock fallback | Check `/dev/video0`, camera permissions, and both MobileNet model files. |
| No labels in `vision.py` | Improve lighting, keep the full object visible, and test a supported class such as `person` or `chair`. |
| FPS below 10 | Reduce other Pi workload and confirm the model is using the native 300x300 input. |
| Gemini is unavailable | Verify `GEMINI_API_KEY`, internet access, and the `.env` location. |
| Console speech instead of JBL audio | Verify the JBL is the Pi's selected output and test `python -X utf8 audio.py`. |
| No `INFORMATIVE` transition | Keep the object visible for at least three ticks with confidence at or above `0.50`. |
| No `URGENT` transition | Move closer until the estimated distance shown in the trace is below `0.60m`. |

---

### CS Senior Track: Fusion, State Machine & Integration

- Owns the core decision loop, rule priority, and final integration.
- Can run the system against mock data, camera distance, or full hardware:
```bash
# 1. Simulated mock mode (tests logic without hardware)
python -X utf8 main.py

# 2. Camera-only distance estimation (uses Logitech webcam bounding box)
python -X utf8 main.py --camera-distance

# 3. Full hardware integration + AI cloud suite
python -X utf8 main.py --camera-distance --gemini --backboard --real
```

---

## 5. Data Contracts & Interface Signatures

Every module conforms strictly to these signatures:

### 1. Vision Module (`vision.py`)
```python
from fusion import Detection

class VisionReader:
    def __init__(self, camera_index: int = 0, confidence_floor: float = 0.40): ...
    def read(self) -> list[Detection]: ...
    def close(self) -> None: ...
```
- **`Detection` Dataclass (defined in `fusion.py`):**
  - `label: str` — Canonical lowercase class name (e.g. `"person"`, `"chair"`). Must match `OBJECT_PRIORITY` keys in `fusion.py`.
  - `confidence: float` — Detection confidence between `0.0` and `1.0`.
  - `bbox: tuple[int, int, int, int]` — Pixel coordinates `(x1, y1, x2, y2)`.
  - `timestamp: float` — `time.monotonic()` seconds.

### 2. Distance Estimation & Serial Reader (`distance_estimator.py` / `serial_reader.py`)
```python
class CameraDistanceEstimator:
    def update_detections(self, detections: list[Detection]) -> None: ...
    def read(self) -> float: ...  # Returns distance in METRES (e.g. 1.45)

class SerialDistanceReader:
    def read(self) -> float: ...  # Returns distance in METRES from Arduino "D:<cm>\n"
    def send_state(self, state_name: str) -> None: ...  # Sends "STATE:<SILENT|INFORMATIVE|URGENT>\n" to Arduino
    def close(self) -> None: ...
```

### 3. Voice Output (`audio.py`)
```python
class AudioOutput:
    def speak(self, text: str) -> None: ...
```
- Must be **non-blocking** (run via background thread) so it never halts the 10 Hz sensing loop.

### 4. Urgent Alert / Haptics (`haptics.py`)
```python
class HapticOutput:
    def buzzer_on(self) -> None: ...
    def buzzer_off(self) -> None: ...
    def cleanup(self) -> None: ...
```
- Both methods are **idempotent** (safe to call repeatedly without side effects).

### 5. Event Logger (`logger.py`)
```python
class EventLogger:
    def log_event(self, frame: SensorFrame, result: FusionResult, state: SystemState) -> None: ...
    def close(self) -> None: ...
```
- Appends CSV records to `guidesense_log.csv` every tick for post-run analysis.

---

## 6. Core Decision Logic & Priority Rules

Every tick (10 Hz), the decision engine executes the following ordered rules in `fusion.py` and `state_machine.py`:

1. **Distance Zone Evaluation (Always Checked First):**
   - $\text{distance} < 0.60\,\text{m} \rightarrow$ **NEAR Zone**
   - $0.60\,\text{m} \le \text{distance} \le 2.00\,\text{m} \rightarrow$ **MID Zone**
   - $\text{distance} > 2.00\,\text{m} \rightarrow$ **FAR Zone**

2. **Near Zone Always Wins (URGENT Fast Path):**
   - If distance is in the NEAR zone, the result is unconditionally **URGENT**.
   - **Never gated by camera vision or persistence.** If someone suddenly steps 40 cm in front of the user, the alarm triggers on tick 1.

3. **Mid-Range Filtering (Triple-Gate Pipeline):**
   To announce an object in the MID zone, it must clear all three gates:
   - **Confidence Gate:** Detection confidence $\ge 0.50$.
   - **Persistence Gate:** Label must appear for **3 consecutive ticks** (kills single-frame detection flicker).
   - **Cooldown Gate:** The same label cannot be announced again within **12.0 seconds** (the system speaks the name *once*, not repeatedly).

4. **Multi-Object Priority Tie-Breaking:**
   If multiple objects clear the gates simultaneously, only the single highest-priority object is announced:
   $$\text{person} > \text{bicycle/motorcycle} > \text{car} > \text{dog} > \text{chair} > \text{table/couch} > \text{other}$$

5. **Buzzer Hysteresis Window (`state_machine.py`):**
   - When distance clears out of the NEAR zone, the urgent alarm stays active for **1.5 seconds** before silencing. This prevents chattering at the 60 cm boundary.

---

## 7. AI Cloud Suite: Gemini + ElevenLabs + Backboard

```
   10 Hz Local Safety Core (0ms Latency)
   [Logitech Camera] ──▶ [fusion.py + state_machine.py] ──▶ Urgent 880Hz Siren / Haptics
                               │
               (When INFORMATIVE action triggers)
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                     AI CLOUD SUITE                          │
   │                                                             │
   │  1. Backboard.io (Durable Spatial Memory & Orchestration)   │
   │     • Stores persistent landmarks & obstacle observations   │
   │     • Semantic vector search (`search_memories`)            │
   │     • Unified LLM reasoning with Gemini 3.8 Flash           │
   │     • Pre-configured with Gemini & ElevenLabs API keys      │
   │                                                             │
   │  2. Google Gemini (Multimodal Vision & Context Reasoning)   │
   │     • Generates natural 1-2 sentence assistive guidance     │
   │     • Accessible directly via REST or routed via Backboard  │
   │                                                             │
   │  3. ElevenLabs (Studio-Quality Ultra-Low Latency Speech)    │
   │     • Speaks guidance in natural, human voice (George)      │
   │     • Zero-dependency PCM-to-WAV playback on JBL speaker    │
   └─────────────────────────────────────────────────────────────┘
```

### Unified Orchestration via Backboard
GuideSense is designed with a **flexible hybrid AI architecture**:
- **Platform-Level Key Routing:** When your Backboard account has Google Gemini and ElevenLabs API keys configured in the Backboard dashboard, Backboard serves as the central intelligent backend. It retrieves spatial memory landmarks via vector similarity (`search_memories`), constructs the assistive prompt, and invokes `gemini-3.8-flash` via `send_message` with memory context.
- **Direct Edge Fallback:** If cloud memory is temporarily unreachable or running offline, GuideSense falls back seamlessly to direct Google Gemini REST API (`gemini_narrator.py`) and local text-to-speech (`pyttsx3` / local voice).
- **Zero Loop Blocking:** All AI calls run in non-blocking background daemon threads. The 10 Hz obstacle detection and urgent siren loop are **never delayed** by network latency.

### Testing AI Modules Standalone
Each AI cloud integration can be verified individually before running the full system:

```bash
# 1. Test Backboard.io Persistent Spatial Memory & Gemini Recall:
python -X utf8 backboard_memory.py

# 2. Test ElevenLabs Natural Voice Playback on JBL Speaker:
python -X utf8 eleven_audio.py

# 3. Test Google Gemini Scene Narrator:
python -X utf8 gemini_narrator.py
```

---

## 8. Operational Modes & CLI Reference

All execution modes are accessible via `main.py`:

```bash
# 1. Full AI Cloud Suite (Webcam Distance + Gemini + ElevenLabs + Backboard Memory)
python -X utf8 main.py --camera-distance --gemini --backboard

# 2. Camera-Only Distance Mode (No physical ultrasonic sensor)
python -X utf8 main.py --camera-distance

# 3. Simulated Mock Mode (Runs immediately with mock sensor streams)
python -X utf8 main.py

# 4. Full Real Hardware Mode (Physical Arduino + Camera + Speaker)
python -X utf8 main.py --real

# 5. Continuous Execution (Run indefinitely until Ctrl+C)
python -X utf8 main.py --camera-distance --forever

# 6. Granular Hardware Flags:
python -X utf8 main.py --real-distance     # Real serial distance only
python -X utf8 main.py --real-vision       # Real camera detector only
python -X utf8 main.py --no-log            # Disable CSV event logging
python -X utf8 main.py --duration 30       # Run for exactly 30 seconds
```

> **Note for Windows:** Always include `-X utf8` to ensure UTF-8 console output without CP1252 errors.

---

## 9. Testing & Quality Assurance Guide

GuideSense contains **62 automated unit tests** covering all nine project brief scenarios, zone boundaries, hysteresis, distance estimation, vision/audio/haptic/logging adapters, Gemini narration, ElevenLabs voice, and Backboard memory.

### Running Tests
```bash
# Run all 62 tests:
pytest
# Or: python -m pytest

# Run only the 9 required competition scenarios:
pytest -m scenario

# Run regression & boundary guard tests:
pytest -m regression

# Run with test coverage report:
pytest --cov=fusion --cov=state_machine --cov=distance_estimator --cov=gemini_narrator --cov=eleven_audio --cov=backboard_memory
```

### Scenario Test Coverage Matrix
| Scenario | Description | Test Function |
|---|---|---|
| **1** | Person at 2 m $\rightarrow$ speak once, then silent | `TestScenario1_PersonAtMidRange` |
| **2** | Chair at 1.5 m $\rightarrow$ speak once, silent after | `TestScenario2_ChairAtMidRange` |
| **3** | Object at 50 cm $\rightarrow$ URGENT regardless of camera | `TestScenario3_ObjectAt50cm` |
| **4** | Close distance, camera sees nothing $\rightarrow$ still URGENT | `TestScenario4_UltrasonicCloseNoCamera` |
| **5** | Camera sees object but distance is far $\rightarrow$ stay silent | `TestScenario5_CameraSeesObjectFarAway` |
| **6** | Detection flickers frame to frame $\rightarrow$ suppressed by persistence gate | `TestScenario6_FlickeringDetections` |
| **7** | Same object stays in view 10+ s $\rightarrow$ announced once only | `TestScenario7_SameObjectTenSeconds` |
| **8** | Two objects at once $\rightarrow$ announce highest priority only | `TestScenario8_MultiObjectPriority` |
| **9** | Person suddenly appears close $\rightarrow$ immediate URGENT (no persistence delay) | `TestScenario9_PersonSuddenlyClose` |
| **+** | Buzzer stays on during 1.5s hysteresis window | `TestUrgentHysteresis` |
| **+** | Voice announcements are idempotent per label | `TestInformativeState` |
| **+** | Monocular distance maps bounding box height to zones | `test_distance_estimator.py` |
| **+** | Gemini scene narrator generates contextual voice guidance | `test_gemini_narrator.py` |
| **+** | ElevenLabs voice synthesis with WAV header & async playback | `test_eleven_audio.py` |
| **+** | Backboard persistent spatial memory & query recall | `test_backboard_memory.py` |

---

## 10. Linting, Code Quality & CI

The repository enforces strict code quality and formatting via [`ruff`](https://docs.astral.sh/ruff/):

```bash
# Check code for style & lint errors:
ruff check .
# Or: python -m ruff check .

# Automatically fix fixable issues:
ruff check --fix .

# Format code:
ruff format .
```

### GitHub Actions CI Workflow (`.github/workflows/ci.yml`)
On every push and pull request to any branch, the CI pipeline automatically:
1. Provisions Python 3.10, 3.11, and 3.12 runners.
2. Installs dependencies from `requirements.txt` via `pip`.
3. Runs the full test suite with coverage reporting.
4. Executes `ruff` linting across the entire codebase.

---

## 11. Troubleshooting & Gotchas

1. **`UnicodeEncodeError: 'charmap' codec can't encode character...`**
   - **Solution:** On Windows PowerShell / Command Prompt, run Python with the `-X utf8` flag: `uv run python -X utf8 main.py`.

2. **Serial Permission Denied on Linux / Raspberry Pi (`/dev/ttyUSB0`)**
   - **Solution:** Add your user to the `dialout` group: `sudo usermod -a -G dialout $USER`, then log out and back in.

3. **No Sound from JBL Speaker on Raspberry Pi**
   - **Solution:** Run `speaker-test -t sine -f 880 -l 1` to verify ALSA output. Check audio device index using `aplay -l`.

4. **ElevenLabs `HTTP 402: Free users cannot use library voices`**
   - **Solution:** Free tier accounts must use pre-made voices (such as `JBFqnCBsd6RMkjVDRZzb` - George) instead of community library voice clones. GuideSense is preconfigured with this voice.

5. **Camera Distance Estimation reads too close / too far**
   - **Solution:** Ensure `DEFAULT_FRAME_HEIGHT = 480` in `distance_estimator.py` matches your camera capture resolution. If capturing at 720p, set `frame_height=720`.
