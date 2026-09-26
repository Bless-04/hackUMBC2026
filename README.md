# GuideSense — Decision Core & Integration Manual

> **Chest-worn assistive navigation device for low-vision users.**  
> Fuses computer vision and proximity data to deliver real-time audio and haptic guidance: staying silent when clear, speaking calm contextual announcements once, and alerting urgently when hazards are near.

---

## 1. System Architecture & Hardware Setup

### Physical Hardware Inventory
- **Compute:** Raspberry Pi (runs main loop, fusion engine, state machine, and computer vision)
- **Vision & Depth:** Logitech Webcam (USB)
- **Audio & Alerts:** JBL Speaker (connected via 3.5mm AUX, USB, or Bluetooth)
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
                  │  (MobileNet-SSD COCO) │        │ (or serial_reader.py) │
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

## 2. Team Track Split & Responsibilities

| Role | Teammate | Modules Owned | Primary Responsibilities |
|---|---|---|---|
| **CS Senior** | Integration Lead | `fusion.py`, `state_machine.py`, `distance_estimator.py`, `main.py`, `tests/` | Decision core logic, priority ranking, persistence & cooldown gates, state transitions, hardware abstraction, CI/CD, mock integration. |
| **CE Freshman** | Hardware & Sensors | `arduino_guidesense/arduino_guidesense.ino`, `serial_reader.py` | Flash Arduino sketch, wire breadboard status LEDs (Green/Yellow/Red) and optional potentiometer dial, physical chest harness assembly. |
| **IT Freshman** | Vision, Audio, Logging | `vision.py`, `audio.py`, `haptics.py`, `logger.py` | Logitech webcam frame capture & MobileNet-SSD detection, pyttsx3 voice on JBL speaker, urgent audio alert loop on JBL speaker, CSV session logging. |

---

## 3. Data Contracts & Interfaces (Critical for Integration)

Each module is strictly decoupled. Teammates must adhere to these exact function signatures:

### 1. Vision Module (`vision.py`)
```python
from fusion import Detection

class VisionReader:
    def __init__(self, camera_index: int = 0, confidence_floor: float = 0.40): ...
    def read(self) -> list[Detection]: ...
    def close(self) -> None: ...
```
- **`Detection` structure:**
  - `label: str` — Lowercase COCO class name (e.g. `"person"`, `"chair"`). Must match `OBJECT_PRIORITY` keys in `fusion.py`.
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
- Must be **non-blocking** (run via a daemon thread or quick async task) so it never halts the 10 Hz sensing loop.
- Emits voice through the connected JBL speaker.

### 4. Urgent Alert / Haptics (`haptics.py`)
```python
class HapticOutput:
    def buzzer_on(self) -> None: ...
    def buzzer_off(self) -> None: ...
    def cleanup(self) -> None: ...
```
- Plays a continuous 880 Hz urgent siren through the JBL speaker while active.
- Both methods are **idempotent** (safe to call multiple times without side effects).

### 5. Event Logger (`logger.py`)
```python
class EventLogger:
    def log_event(self, frame: SensorFrame, result: FusionResult, state: SystemState) -> None: ...
    def close(self) -> None: ...
```
- Appends CSV records to `guidesense_log.csv` every tick for post-run analysis and demo verification.

---

## 4. Arduino Breadboard Controller (`arduino_guidesense.ino`)

The CE freshman flashes the sketch located in `arduino_guidesense/arduino_guidesense.ino`:

### Breadboard Wiring
| Arduino Pin | Component | Purpose |
|---|---|---|
| **Pin 2** | Green LED (with 220Ω resistor to GND) | Lights when system state is `SILENT` (clear path) |
| **Pin 3** | Yellow LED (with 220Ω resistor to GND) | Lights when system state is `INFORMATIVE` (mid-range object) |
| **Pin 4** | Red LED (with 220Ω resistor to GND) | Lights when system state is `URGENT` (proximity hazard) |
| **Pin A0** | Potentiometer center wiper pin | Optional: manual distance dial (sends `D:<cm>\n` over serial) |
| **5V / GND** | Breadboard power rails | Power rails for LEDs and potentiometer |

---

## 5. Core Decision Logic (`fusion.py` & `state_machine.py`)

Every tick (10 Hz), the decision engine executes the following ordered rules:

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
   - When distance clears out of the NEAR zone, the urgent alarm stays active for **1.5 seconds** before silencing. This prevents chattering/flickering at the 60 cm boundary.

---

## 6. Running GuideSense

GuideSense uses [`uv`](https://docs.astral.sh/uv/) for Python packaging and virtual environments.

### Install & Synchronize
```bash
# Clone the repository
git clone https://github.com/your-org/hackUMBC2026.git
cd hackUMBC2026

# Install development environment and dependencies
uv sync --dev
```

### Operational Modes

#### 1. Simulated / Mock Mode (Runs immediately without any hardware)
```bash
uv run python -X utf8 main.py
```
*Simulates a person approaching from 4.0m to 0.4m, testing the full SILENT $\rightarrow$ INFORMATIVE $\rightarrow$ URGENT $\rightarrow$ SILENT pipeline.*

#### 2. Webcam Distance Mode (Logitech Webcam Connected, No Ultrasonic Sensor)
```bash
uv run python -X utf8 main.py --camera-distance
```
*Uses the Logitech webcam for both object detection and monocular distance estimation.*

#### 3. Full Real Hardware Mode (Arduino Breadboard + Webcam + JBL Speaker)
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
uv run python -X utf8 main.py --real
```

#### 4. Additional CLI Options
```bash
uv run python -X utf8 main.py --camera-distance --forever   # Run continuously until Ctrl+C
uv run python -X utf8 main.py --real-distance              # Only real serial reader, mock vision
uv run python -X utf8 main.py --real-vision                # Only real camera, mock distance
uv run python -X utf8 main.py --no-log                     # Disable CSV session logging
uv run python -X utf8 main.py --duration 30                # Run for 30 seconds
```

> **Note for Windows:** Always include `-X utf8` to ensure UTF-8 console output.

---

## 7. Testing & Verification

GuideSense includes 46 unit tests covering all 9 project brief scenarios, zone boundaries, hysteresis, and distance estimation:

```bash
# Run all tests
uv run pytest

# Run only the 9 required competition scenarios
uv run pytest -m scenario

# Run regression & boundary guard tests
uv run pytest -m regression

# Run with test coverage report
uv run pytest --cov=fusion --cov=state_machine --cov=distance_estimator
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

---

## 8. Continuous Integration (CI)

A GitHub Actions workflow is active under `.github/workflows/ci.yml`. On every push and pull request, it:
1. Provisions Python 3.10, 3.11, and 3.12 runners.
2. Installs `uv` and synchronizes dependencies.
3. Executes `pytest` with coverage report generation.
4. Runs `ruff` linting across all source and test files.
