# GuideSense

GuideSense is a camera based assistive navigation prototype. A Logitech USB
webcam supplies the live image. OpenCV and MobileNet SSD detect objects, and
their apparent size supplies a rough distance estimate. The app shows the live
view, direction, estimated distance, and guidance in a local browser dashboard.

The computer running GuideSense provides processing and, if enabled, speech and
urgent tones through its normal audio output. No separate controller, distance
sensor, or dedicated alert device is part of this project.

## Run the dashboard

Install Python 3.10 or newer and connect the Logitech camera by USB. From the
repository root, create a virtual environment and install the project:

| Platform | Setup | Launch |
| --- | --- | --- |
| Windows PowerShell | `py -m venv .venv` then `.\.venv\Scripts\python.exe -m pip install -e .` | `.\.venv\Scripts\python.exe -m ui` |
| macOS / Linux | `python3 -m venv .venv` then `.venv/bin/python -m pip install -e .` | `.venv/bin/python -m ui` |

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) on the same computer. The
server listens only on localhost. The dashboard starts in a simulated demo;
choose **Configure → Live camera**, select index **0** or **1**, save, and then
select **Start session** to use the actual webcam. End the session before another
app opens the camera. The camera never needs browser permission because OpenCV
reads it in the Python process.

The browser UI, session controls, event export, and troubleshooting details are
documented in [ui/README.md](ui/README.md).

## Camera only demonstration checklist

1. Connect the camera to the computer and close video meeting or camera apps.
   If index 0 opens the built in webcam, choose index 1 in Configuration.
2. Start a **Live camera** session. Success: the badge shows **LIVE**, the live
   image appears, the elapsed timer advances, and an object such as a person,
   chair, or bottle is outlined with a label and confidence value.
3. With a helper supervising, change how much of the image that object fills.
   Success: the displayed estimated distance and near/mid/far zone change. A
   mid range, confirmed object produces guidance; a very large near detection
   causes an urgent state. The session timeline records these changes.
4. Optionally enable **Voice & urgent alerts** to use the computer's selected
   speakers or headphones. You can also enable Gemini and Backboard if you have
   configured their API keys. Each integration is off by default.
5. Select **End session**. Success: the status returns to standby and the camera
   is released. Select **Session activity → Export CSV** for an event timeline.

Camera distance is a bounding box heuristic, not measured depth. Image framing,
camera angle, and object size affect the estimate. Test in a controlled space;
do not rely on the prototype to avoid hazards while walking.

## Command line mode

The same sensing pipeline can run without the browser:

```sh
python main.py                            # simulated data, 10 seconds
python main.py --camera-distance --camera 0 --preview --duration 30
python main.py --real --camera 0 --forever # camera plus computer audio
```

Use your virtual environment's Python executable if `python` is not mapped to
it. `--real-vision` is an alias for `--camera-distance`; `--gui` is an alias for
`--preview`. `--audio` enables computer speech and urgent tones independently
of the camera flag. `--gemini`, `--backboard`, and `--no-log` control the
optional integrations and CSV logging. A selected live camera fails clearly if
it cannot open instead of showing simulated detections.

## How it works

```text
Logitech webcam → vision.py (OpenCV + MobileNet SSD) → detections
                                           ↓
                         distance_estimator.py → estimated metres
                                           ↓
                          fusion.py → state_machine.py
                                           ↓
                    ui/ dashboard, computer audio, CSV log
```

`vision.py` uses the bundled `MobileNetSSD_deploy.prototxt` and
`MobileNetSSD_deploy.caffemodel`. The camera supplies both the displayed frames
and the detections used by the decision engine. `fusion.py` applies the same
near, mid, and far rules on every platform. `audio.py` speaks guidance through
ElevenLabs when configured, otherwise through local text to speech where
available. `haptics.py` preserves the state machine alert interface but plays
an urgent tone through the computer audio output. It does not drive a physical
vibration device.

For optional cloud features, set `GEMINI_API_KEY`, `ELEVEN_LABS_API_KEY`, and/or
`BACKBOARD_API_KEY` as environment variables or in a local `.env` file. The
`.env` file is ignored by Git. Gemini may receive a camera image, Backboard
receives observation summaries, and ElevenLabs receives guidance text when you
enable the corresponding feature. Local detection runs without these keys.

## Platform notes

- Windows plays generated WAV audio through the system audio API.
- macOS uses its built in `afplay` utility.
- Linux uses `paplay` or `aplay` when available. Local text to speech may also
  need the platform's normal speech engine installed. If audio playback is
  unavailable, the app continues showing visual guidance and logs the alert.
- All platforms use the same Python camera, detection, distance, fusion, UI,
  and logging code. The webcam must be accessible to the operating system.

## Verify the code

```sh
python -m pytest
python -m ruff check .
```

The automated suite mocks camera and audio hardware. For a real camera test,
follow the live checklist above and watch the dashboard for actual camera
frames, detection boxes, changing distance, and timeline events.
