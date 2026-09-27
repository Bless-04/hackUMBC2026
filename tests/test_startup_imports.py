"""Exercise entry points without pytest's legacy import aliases installed."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_camera_session_starts_in_fresh_interpreter(tmp_path):
    script = textwrap.dedent("""
        import sys
        import time
        from pathlib import Path
        from unittest.mock import Mock, patch

        import numpy as np
        from ui import runtime

        assert 'core._compat' not in sys.modules
        runtime.SESSION_DIR = Path(sys.argv[1])
        reader = Mock(latest_frame=None)
        def read():
            reader.latest_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            return []
        reader.read.side_effect = read
        controller = runtime.SessionController()

        with (
            patch('vision.vision.VisionReader', return_value=reader),
            patch('hardware.audio.AudioOutput'),
            patch('hardware.haptics.HapticOutput'),
            patch('services.gemini_narrator.GeminiNarrator'),
            patch('services.backboard_memory.BackboardMemory'),
        ):
            try:
                controller.start(dict(mode='camera', voice=True, gemini=True,
                                      backboard=True, logging=True))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    state = controller.snapshot()
                    if state['status'] == 'error' or state['ticks'] >= 3:
                        break
                    time.sleep(.01)
                assert state['status'] == 'running', state
                assert state['ticks'] >= 3, state
                assert state['has_frame'], state
            finally:
                controller.shutdown()

            reader.close.assert_called_once()
            assert list(runtime.SESSION_DIR.glob('session-*.csv'))

            # Importing the CLI must not be necessary to make the UI work.
            import main
            main.run(use_camera=True, enable_audio=True, enable_gemini=True,
                     enable_backboard=True, enable_logging=False,
                     duration_sec=.02, verbose=False)
        assert 'core._compat' not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
