"""
hud.py — GuideSense Live Visual HUD Overlay
============================================
Real-time OpenCV-based Heads-Up Display (HUD) for GuideSense.

Overlays on live webcam frames (or simulated canvas in mock mode):
  1. Bounding boxes around detected objects.
  2. Object class label.
  3. Detection confidence percentage.
  4. Estimated distance in meters (per object and fused sensor distance).
  5. Distance zone: FAR (teal), MID (amber), NEAR (bright red).
  6. Direction: LEFT, CENTER, RIGHT with vertical zone dividers.
  7. Current GuideSense system state: SILENT (green), INFORMATIVE (amber), URGENT (red).
  8. Enabled system features: Camera, Arduino, Gemini, Backboard, Logging.

Visual behavior:
  - FAR objects: Non-alarming cool/teal visual treatment.
  - MID objects: Distinct amber/yellow visual treatment with badge.
  - NEAR objects: Highly obvious bright red treatment with warning badge.
  - Arbitrary camera resolutions supported via proportional scaling.

Keyboard controls:
  - 'q' or ESC : Quit application cleanly.
  - 'g'        : Toggle HUD information overlay.
  - SPACE      : Pause / resume live stream.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from distance import estimate_distance_m, get_real_height_cm
from fusion import (
    MID_THRESHOLD_M,
    NEAR_THRESHOLD_M,
    Detection,
    Direction,
    FusionAction,
    FusionResult,
    SensorFrame,
    Zone,
    compute_direction,
)
from state_machine import SystemState


# ---------------------------------------------------------------------------
# Color Palette (BGR for OpenCV)
# ---------------------------------------------------------------------------

COLOR_NEAR       = (0, 0, 230)      # Bright Red
COLOR_MID        = (0, 200, 255)    # Amber / Yellow
COLOR_FAR        = (220, 180, 40)   # Teal / Cyan-Blue
COLOR_NEUTRAL    = (180, 180, 180)  # Light Grey
COLOR_BG_DARK    = (20, 20, 20)     # Dark background for status bars
COLOR_WHITE      = (255, 255, 255)  # Pure White
COLOR_BLACK      = (0, 0, 0)        # Pure Black
COLOR_GREEN      = (40, 180, 40)    # Green for SILENT state
COLOR_AMBER      = (0, 165, 255)    # Amber for INFORMATIVE state
COLOR_URGENT     = (0, 0, 255)      # Flashing Red for URGENT state


# ---------------------------------------------------------------------------
# GuideSenseHUD
# ---------------------------------------------------------------------------

class GuideSenseHUD:
    """
    Live visual HUD overlay renderer for GuideSense.
    Operates non-blockingly inside the main sensor/fusion loop.
    """

    def __init__(
        self,
        window_name: str = "GuideSense — Live Navigation HUD",
        features: Optional[dict[str, str]] = None,
    ) -> None:
        self.window_name = window_name
        self.features: dict[str, str] = features or {
            "Camera": "Active",
            "Arduino": "Mock",
            "Gemini": "Off",
            "Backboard": "Off",
            "Logging": "Active",
        }
        self.show_hud: bool = True
        self.is_paused: bool = False
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_time: float = time.monotonic()
        self._window_created: bool = False

    def _ensure_window(self) -> None:
        if not self._window_created:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_created = True
            except Exception:
                pass

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        frame: SensorFrame,
        result: FusionResult,
        state: SystemState,
    ) -> np.ndarray:
        """
        Draws all HUD telemetry, bounding boxes, spatial lines, and status badges onto a frame copy.
        Resolution-independent: scales dimensions and fonts to the frame's actual resolution.
        """
        disp = frame_bgr.copy()
        h, w = disp.shape[:2]

        # Resolution scale factor normalized to 640x480
        scale = max(0.6, min(w / 640.0, h / 480.0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # -------------------------------------------------------------------
        # 1. Spatial Zone Guideline Dividers (LEFT < 0.33, CENTER, RIGHT > 0.66)
        # -------------------------------------------------------------------
        if self.show_hud:
            x_left = int(w * 0.33)
            x_right = int(w * 0.66)

            # Subtle vertical division lines
            overlay = disp.copy()
            cv2.line(overlay, (x_left, 45), (x_left, h - 35), (100, 100, 100), 1)
            cv2.line(overlay, (x_right, 45), (x_right, h - 35), (100, 100, 100), 1)

            # Spatial Zone header labels
            zone_font_scale = 0.45 * scale
            cv2.putText(overlay, "LEFT ZONE (<33%)", (int(w * 0.08), h - 45), font, zone_font_scale, (120, 120, 120), 1)
            cv2.putText(overlay, "CENTER ZONE (33-66%)", (int(w * 0.40), h - 45), font, zone_font_scale, (120, 120, 120), 1)
            cv2.putText(overlay, "RIGHT ZONE (>66%)", (int(w * 0.74), h - 45), font, zone_font_scale, (120, 120, 120), 1)

            cv2.addWeighted(overlay, 0.4, disp, 0.6, 0, disp)

        # -------------------------------------------------------------------
        # 2. Bounding Boxes, Labels, Distance, and Spatial Badges
        # -------------------------------------------------------------------
        for det in frame.detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            bbox_h = abs(y2 - y1)

            # Calculate individual object distance from bounding-box geometry
            real_h = get_real_height_cm(det.label)
            obj_dist_m = estimate_distance_m(bbox_h, real_h) if bbox_h > 0 else 4.0

            # Determine object-level distance zone
            if obj_dist_m < NEAR_THRESHOLD_M:
                obj_zone = Zone.NEAR
                box_color = COLOR_NEAR
                thickness = max(2, int(3 * scale))
            elif obj_dist_m <= MID_THRESHOLD_M:
                obj_zone = Zone.MID
                box_color = COLOR_MID
                thickness = max(2, int(2 * scale))
            else:
                obj_zone = Zone.FAR
                box_color = COLOR_FAR
                thickness = max(1, int(1.5 * scale))

            # Direction
            direction = det.direction or compute_direction(det.bbox, w)

            # Draw bounding box
            cv2.rectangle(disp, (x1, y1), (x2, y2), box_color, thickness)

            # Corner brackets for distinct / urgent visual focus
            corner_len = min(20, max(8, int((x2 - x1) * 0.15)))
            cv2.line(disp, (x1, y1), (x1 + corner_len, y1), box_color, thickness + 1)
            cv2.line(disp, (x1, y1), (x1, y1 + corner_len), box_color, thickness + 1)
            cv2.line(disp, (x2, y1), (x2 - corner_len, y1), box_color, thickness + 1)
            cv2.line(disp, (x2, y1), (x2, y1 + corner_len), box_color, thickness + 1)

            if self.show_hud:
                # Text badge: Label, Confidence, Distance, Zone, Direction
                badge_text = (
                    f"{det.label.upper()} {int(det.confidence * 100)}% | "
                    f"{obj_dist_m:.2f}m [{obj_zone.name}] | {direction.name}"
                )
                text_scale = 0.45 * scale
                (tw, th), baseline = cv2.getTextSize(badge_text, font, text_scale, 1)

                badge_y1 = max(0, y1 - th - 10)
                badge_y2 = y1 if y1 > (th + 10) else (y1 + th + 10)

                # Solid badge background
                cv2.rectangle(
                    disp,
                    (x1, badge_y1),
                    (x1 + tw + 10, badge_y2),
                    box_color,
                    -1,
                )
                # Contrasting text
                text_color = COLOR_WHITE if obj_zone != Zone.MID else COLOR_BLACK
                text_pos_y = (y1 - 5) if y1 > (th + 10) else (badge_y2 - 5)
                cv2.putText(
                    disp,
                    badge_text,
                    (x1 + 5, text_pos_y),
                    font,
                    text_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        # -------------------------------------------------------------------
        # 3. Top System Telemetry Bar (State, Distance, FPS)
        # -------------------------------------------------------------------
        top_bar_h = int(45 * scale)
        top_overlay = disp.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, top_bar_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(top_overlay, 0.85, disp, 0.15, 0, disp)

        # Left: Title & FPS
        fps_text = f"GuideSense HUD | {self._fps:.1f} FPS"
        cv2.putText(disp, fps_text, (int(12 * scale), int(28 * scale)), font, 0.55 * scale, COLOR_WHITE, 1, cv2.LINE_AA)

        # Center: System State Badge
        state_name = state.name
        if state == SystemState.URGENT:
            state_bg = COLOR_URGENT
            state_text_color = COLOR_WHITE
            state_label = f" STATE: URGENT (HAZARD NEAR) "
        elif state == SystemState.INFORMATIVE:
            state_bg = COLOR_AMBER
            state_text_color = COLOR_BLACK
            active_lbl = f" - {result.label.upper()}" if result.label else ""
            state_label = f" STATE: INFORMATIVE{active_lbl} "
        else:  # SILENT
            state_bg = COLOR_GREEN
            state_text_color = COLOR_WHITE
            state_label = " STATE: SILENT (CLEAR) "

        (stw, sth), _ = cv2.getTextSize(state_label, font, 0.52 * scale, 2)
        state_x1 = int((w - stw) / 2)
        state_y1 = int(7 * scale)
        cv2.rectangle(disp, (state_x1 - 6, state_y1), (state_x1 + stw + 6, state_y1 + sth + int(14 * scale)), state_bg, -1)
        cv2.putText(
            disp,
            state_label,
            (state_x1, state_y1 + sth + int(6 * scale)),
            font,
            0.52 * scale,
            state_text_color,
            2,
            cv2.LINE_AA,
        )

        # Right: Fused Sensor Distance & Zone
        fused_zone = result.zone.name if result.zone else ("NEAR" if frame.distance_m < NEAR_THRESHOLD_M else ("MID" if frame.distance_m <= MID_THRESHOLD_M else "FAR"))
        dist_text = f"DIST: {frame.distance_m:.2f}m [{fused_zone}]"
        (dtw, _), _ = cv2.getTextSize(dist_text, font, 0.52 * scale, 1)
        cv2.putText(
            disp,
            dist_text,
            (w - dtw - int(15 * scale), int(28 * scale)),
            font,
            0.52 * scale,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # 4. Bottom System Status Bar (Active Subsystems & Controls)
        # -------------------------------------------------------------------
        bot_bar_h = int(32 * scale)
        bot_overlay = disp.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_bar_h), (w, h), COLOR_BG_DARK, -1)
        cv2.addWeighted(bot_overlay, 0.85, disp, 0.15, 0, disp)

        # Left: Enabled features status
        feat_str = (
            f"CAM:{self.features.get('Camera', 'Active')}  |  "
            f"ARDUINO:{self.features.get('Arduino', 'Mock')}  |  "
            f"GEMINI:{self.features.get('Gemini', 'Off')}  |  "
            f"BACKBOARD:{self.features.get('Backboard', 'Off')}  |  "
            f"LOG:{self.features.get('Logging', 'Active')}"
        )
        cv2.putText(
            disp,
            feat_str,
            (int(10 * scale), h - int(10 * scale)),
            font,
            0.42 * scale,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Right: Controls hint
        controls_text = "[Q] Quit  [G] HUD  [SPACE] Pause"
        (ctw, _), _ = cv2.getTextSize(controls_text, font, 0.42 * scale, 1)
        cv2.putText(
            disp,
            controls_text,
            (w - ctw - int(10 * scale), h - int(10 * scale)),
            font,
            0.42 * scale,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # -------------------------------------------------------------------
        # 5. Paused Overlay (if active)
        # -------------------------------------------------------------------
        if self.is_paused:
            pause_text = "PAUSED — PRESS [SPACE] TO RESUME"
            (ptw, pth), _ = cv2.getTextSize(pause_text, font, 0.7 * scale, 2)
            px = int((w - ptw) / 2)
            py = int(h / 2)
            cv2.rectangle(disp, (px - 15, py - pth - 15), (px + ptw + 15, py + 15), (0, 0, 0), -1)
            cv2.rectangle(disp, (px - 15, py - pth - 15), (px + ptw + 15, py + 15), COLOR_AMBER, 2)
            cv2.putText(disp, pause_text, (px, py), font, 0.7 * scale, COLOR_WHITE, 2, cv2.LINE_AA)

        return disp

    def render(
        self,
        frame_img: Optional[np.ndarray],
        frame: SensorFrame,
        result: FusionResult,
        state: SystemState,
    ) -> str:
        """
        Renders HUD onto the latest camera frame (or synthetic canvas), displays it,
        and polls keyboard events non-blockingly.

        Returns:
            "QUIT" if user requested exit ('q' or ESC),
            "CONTINUE" otherwise.
        """
        self._ensure_window()

        # Update FPS calculation
        self._frame_count += 1
        now = time.monotonic()
        dt = now - self._last_fps_time
        if dt >= 0.5:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._last_fps_time = now

        # Use synthetic frame if webcam frame is None (e.g. mock vision reader)
        if frame_img is None:
            frame_img = self._create_synthetic_canvas(frame.frame_width or 640, 480)

        disp = self.draw_hud(frame_img, frame, result, state)

        try:
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                return "QUIT"
            elif key in (ord('g'), ord('G')):
                self.show_hud = not self.show_hud
            elif key == ord(' '):
                self.is_paused = not self.is_paused
                # Pause handling loop
                while self.is_paused:
                    disp_paused = self.draw_hud(frame_img, frame, result, state)
                    cv2.imshow(self.window_name, disp_paused)
                    pause_key = cv2.waitKey(30) & 0xFF
                    if pause_key in (ord('q'), ord('Q'), 27):
                        return "QUIT"
                    elif pause_key == ord(' '):
                        self.is_paused = False
                    elif pause_key in (ord('g'), ord('G')):
                        self.show_hud = not self.show_hud

        except Exception as e:
            print(f"[GuideSenseHUD] Display render warning: {e}")

        return "CONTINUE"

    def _create_synthetic_canvas(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Generates a clean synthetic camera canvas with grid lines for mock mode."""
        canvas = np.full((height, width, 3), 35, dtype=np.uint8)
        # Subtle grid pattern
        for y in range(0, height, 40):
            cv2.line(canvas, (0, y), (width, y), (45, 45, 45), 1)
        for x in range(0, width, 40):
            cv2.line(canvas, (x, 0), (x, height), (45, 45, 45), 1)

        cv2.putText(
            canvas,
            "SIMULATED CAMERA STREAM",
            (int(width * 0.32), int(height * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def close(self) -> None:
        """Closes OpenCV HUD window cleanly."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
