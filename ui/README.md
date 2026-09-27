# GuideSense dashboard

A local browser interface for the existing GuideSense application. All UI code
lives here; `vision.py` and the original CLI/OpenCV HUD remain unchanged.
No Node.js, frontend build, external fonts, or new runtime dependencies are required.

## Launch

From the repository root, with the project's Python environment activated:

```sh
python -m ui
```

Open **http://127.0.0.1:8765** in a current browser. To use another port:

```sh
python -m ui --port 8766
```

On Windows without activating the environment, run `.venv\Scripts\python.exe -m ui`.
On Raspberry Pi, run the server and browser on the Pi. The server binds to
loopback only: it is not exposed to other devices on your network.

## Try the demo first

Select **Start session**. Every 18 seconds a simulated person approaches and
moves away. The distance, bounding box, object table, guidance, and awareness state
update using the real fusion engine and state machine. Demo imagery is explicitly
labeled as simulated; it is not a real camera feed or model inference.

Success means the timer advances, the person appears with 94% simulated confidence,
the awareness state becomes informative/urgent as the person approaches, and
**Session activity** records those changes. **End session** should return the
dashboard to standby. Demo mode never opens hardware, plays sound, or calls cloud APIs.

## Test your Logitech camera and AI integrations

1. Connect the camera to the computer running the server. Close other camera apps
   and any running `main.py`/OpenCV session so they do not compete for the webcam.
2. Open **Configure → Live camera**. Start with camera index **0**; use **1** if the
   computer's built-in camera occupies index 0. Save, then start the session.
3. Present a supported object such as a person, chair, or bottle in good light.
   Success: the badge says **LIVE**, the actual camera image appears, detection boxes
   and confidence values update, and the elapsed time advances. The displayed FPS
   is the sensing-loop rate (target 10 Hz), not a standalone model benchmark.
4. Test at a desk with supervision. Change the apparent object size in the frame.
   The distance estimate and near/mid/far indicators should change. Monocular
   distance is a rough heuristic, **not calibrated ranging or a safety guarantee**.
5. End the session before changing settings. For sound, select the JBL speaker as
   your operating system's output and enable **Voice & urgent alerts**. It uses
   the existing audio/haptic adapters, including ElevenLabs or local voice fallback.
6. Optionally enable **Gemini scene guidance** and/or **Backboard memory** with keys
   configured in the repository's `.env` or environment. Gemini receives camera
   images; Backboard receives object/distance observations. Voice may send guidance
   text to ElevenLabs when configured. These features are off by default.
7. A service marked **Configured** has a credential, not a verified connection.
   For Gemini, success is an actual Gemini-attributed announcement in the guidance
   card/timeline. For Backboard, verify successful writes in its service or terminal;
   the UI does not independently verify persistence. Core local detection remains
   available when optional cloud services are disabled.

An Arduino, breadboards, and additional wires are **not needed** for this camera-only
dashboard test. It uses camera-estimated distance, not serial distance input.
Physical haptic/Arduino wiring continues to be configured in the existing adapters;
this interface does not add a wiring or serial-port configuration screen.

## Activity, logs, and shutdown

- **Session activity → Export CSV** downloads the latest 200 session events.
  Starting a new session clears the in-memory timeline.
- **Save session log** records every sensing tick using the existing logger in
  `ui/sessions/session-<timestamp>.csv`. This directory is Git-ignored. Video frames
  are served from memory, not saved to disk.
- **End session** releases the camera and audio/alert outputs. Closing the browser
  tab does **not** stop the session; use End session or Ctrl+C in the server terminal.
- A missing camera/model is shown as an error, never silently replaced with demo
  data. If a fresh camera frame is unavailable, the feed shows **NO FRAME** and the
  distance is blank. Read terminal output for setup failures.
- Configure is locked while a session is running or stopping. The **F** key expands
  the camera view; Escape leaves fullscreen or closes dialogs. Controls support
  keyboard navigation, visible focus, responsive layouts, and reduced-motion settings.

## Files and verification

```text
ui/
  __main__.py       python -m ui entry point
  server.py         loopback HTTP API and explicit static-file allowlist
  runtime.py        threaded sessions and existing-module integration
  static/
    index.html     dashboard and dialogs
    app.css        responsive sage/forest visual design
    app.js         session controls, polling, overlays, and CSV download
    scene.svg      clearly labeled illustrated demo environment
    mark.svg       GuideSense mark
  sessions/        optional local CSV recordings (ignored by Git)
```

Run `python -m pytest tests/test_ui.py` for configuration, lifecycle, simulated
camera, cleanup, logging, and HTTP security tests. Run `python -m pytest` for
the full suite. Physical camera/speaker and cloud-service testing require your
hardware and credentials and are not covered by the mocked tests.
