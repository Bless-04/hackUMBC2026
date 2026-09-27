"""
hud.py — GuideSense Live Visual HUD Overlay
============================================
Professional, real-time OpenCV-based Heads-Up Display (HUD) for GuideSense.

Overlays on live webcam frames (or simulated canvas in mock mode):
  1. High-tech smoothed bounding boxes with corner reticles and translucent target fill.
  2. Temporal smoothing (EMA) and fade transitions to eliminate detection jitter and flicker.
  3. Dynamic zone color-coding: FAR (Cyber Cyan), MID (Amber Gold), NEAR (Crimson Alert).
  4. Object classification label, confidence gauge, and metric distance estimation.
  5. Directional spatial lane indicators (LEFT / CENTER / RIGHT) with active lane highlights.
  6. Floating frosted glass telemetry bars for system state, distance meter, and active features.
  7. Smooth visual alert pulsing for urgent hazards without blocking the 10 Hz sensing loop.
  8. Resolution-independent rendering with anti-aliased geometry and typography.

Keyboard controls:
  - 'q' or ESC : Quit application cleanly.
  - 'g'        : Toggle HUD information overlay.
  - SPACE      : Pause / resume live stream.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from distance import estimate_distance_m, get_real_height_cm
from fusion import (
    MID_THRESHOLD_M,
    NEAR_THRESHOLD_M,
    Direction,
    FusionResult,
    SensorFrame,
    Zone,
    compute_direction,
)
from state_machine import SystemState

# ---------------------------------------------------------------------------
# Visual Theme & Color Palette (BGR for OpenCV)
# ---------------------------------------------------------------------------

class HUDTheme:
    """Modern dark theme color palette with cybernetic / assistive telemetry styling."""
    # Backgrounds & Panels
    PANEL_BG        = (18, 20, 26)      # Deep obsidian slate
    PANEL_BORDER    = (60, 68, 82)      # Titanium border
    PANEL_BORDER_HI = (110, 130, 160)   # Highlight border
    GRID_LINE       = (35, 40, 50)      # Subtle grid line

    # Typography
    TEXT_PRIMARY    = (248, 250, 252)   # Crisp white
    TEXT_SECONDARY  = (165, 175, 192)   # Cool silver grey
    TEXT_MUTED      = (95, 105, 120)    # Muted dark grey
    TEXT_DARK       = (15, 18, 22)      # For high-contrast on light badges

    # Zones & Alerts
    COLOR_NEAR      = (45, 45, 255)     # Neon Crimson / Hazard Red (< 0.60m)
    COLOR_MID       = (30, 180, 255)    # Vibrant Amber / Gold (0.60m - 2.00m)
    COLOR_FAR       = (230, 175, 40)    # Cyber Cyan / Ambient Teal (> 2.00m)

    # System States
    STATE_SILENT    = (65, 200, 75)     # Emerald Green (Clear Path)
    STATE_INFORMATIVE = (30, 180, 255)  # Amber
    STATE_URGENT    = (45, 45, 255)     # Neon Crimson (Urgent Alert)

    # Accents
    ACCENT_CYAN     = (240, 190, 50)    # Azure Cyan
    ACCENT_BLUE     = (240, 140, 20)    # Deep Tech Blue
    WHITE           = (255, 255, 255)
    BLACK           = (0, 0, 0)

# Backward-compatible top-level constants for existing test suites
COLOR_NEAR    = HUDTheme.COLOR_NEAR
COLOR_MID     = HUDTheme.COLOR_MID
COLOR_FAR     = HUDTheme.COLOR_FAR
COLOR_NEUTRAL = HUDTheme.TEXT_SECONDARY
COLOR_BG_DARK = HUDTheme.PANEL_BG
COLOR_WHITE   = HUDTheme.WHITE
COLOR_BLACK   = HUDTheme.BLACK
COLOR_GREEN   = HUDTheme.STATE_SILENT
COLOR_AMBER   = HUDTheme.STATE_INFORMATIVE
COLOR_URGENT  = HUDTheme.STATE_URGENT


# ---------------------------------------------------------------------------
# Smooth Multi-Object Tracker (EMA Box Interpolation & Anti-Flicker)
# ---------------------------------------------------------------------------

@dataclass
class TrackedBox:
    """Tracks bounding box coordinates across frames with exponential smoothing."""
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    distance_m: float
    zone: Zone
    direction: Direction
    alpha: float = 1.0          # For smooth fade-in / fade-out
    missed_frames: int = 0
    updated_at: float = 0.0


class DetectionTracker:
    """
    Lightweight, high-performance temporal tracker for bounding boxes.
    Applies Exponential Moving Average (EMA) to prevent coordinate jitter
    and handles short-term temporal continuity to eliminate flickering.
    """

    def __init__(self, smoothing_factor: float = 0.60, max_missed: int = 2) -> None:
        self.smoothing = smoothing_factor
        self.max_missed = max_missed
        self.tracked_boxes: list[TrackedBox] = []

    def update(
        self,
        detections: list[Any],
        frame_w: int,
        frame_h: int,
    ) -> list[TrackedBox]:
        now = time.monotonic()
        updated_boxes: list[TrackedBox] = []
        matched_indices: set[int] = set()

        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            dx1, dy1 = max(0.0, float(dx1)), max(0.0, float(dy1))
            dx2, dy2 = min(float(frame_w), float(dx2)), min(float(frame_h), float(dy2))
            bbox_h = abs(dy2 - dy1)

            # Compute metric distance & zone
            real_h = get_real_height_cm(det.label)
            dist_m = estimate_distance_m(bbox_h, real_h) if bbox_h > 0 else 4.0
            if dist_m < NEAR_THRESHOLD_M:
                zone = Zone.NEAR
            elif dist_m <= MID_THRESHOLD_M:
                zone = Zone.MID
            else:
                zone = Zone.FAR

            direction = det.direction or compute_direction(det.bbox, frame_w)
            det_cx = (dx1 + dx2) / 2.0
            det_cy = (dy1 + dy2) / 2.0

            # Match against existing tracked box (same label + nearest center)
            best_match_idx = -1
            min_dist = float("inf")
            for i, tb in enumerate(self.tracked_boxes):
                if i in matched_indices or tb.label != det.label:
                    continue
                tcx = (tb.x1 + tb.x2) / 2.0
                tcy = (tb.y1 + tb.y2) / 2.0
                dist_px = math.hypot(det_cx - tcx, det_cy - tcy)
                if dist_px < min_dist and dist_px < (frame_w * 0.35):
                    min_dist = dist_px
                    best_match_idx = i

            if best_match_idx >= 0:
                # Update existing box with EMA smoothing
                matched_indices.add(best_match_idx)
                tb = self.tracked_boxes[best_match_idx]
                s = self.smoothing
                tb.x1 = s * dx1 + (1.0 - s) * tb.x1
                tb.y1 = s * dy1 + (1.0 - s) * tb.y1
                tb.x2 = s * dx2 + (1.0 - s) * tb.x2
                tb.y2 = s * dy2 + (1.0 - s) * tb.y2
                tb.confidence = s * float(det.confidence) + (1.0 - s) * tb.confidence
                tb.distance_m = s * dist_m + (1.0 - s) * tb.distance_m
                tb.zone = zone
                tb.direction = direction
                tb.alpha = min(1.0, tb.alpha + 0.35)
                tb.missed_frames = 0
                tb.updated_at = now
                updated_boxes.append(tb)
            else:
                # Spawn new tracked box with initial fade-in
                new_box = TrackedBox(
                    label=det.label,
                    confidence=float(det.confidence),
                    x1=dx1,
                    y1=dy1,
                    x2=dx2,
                    y2=dy2,
                    distance_m=dist_m,
                    zone=zone,
                    direction=direction,
                    alpha=0.65,
                    missed_frames=0,
                    updated_at=now,
                )
                updated_boxes.append(new_box)

        # Handle unmatched existing boxes (graceful fade-out over missed frames)
        for i, tb in enumerate(self.tracked_boxes):
            if i not in matched_indices:
                tb.missed_frames += 1
                tb.alpha = max(0.0, tb.alpha - 0.40)
                if tb.missed_frames <= self.max_missed and tb.alpha > 0.1:
                    updated_boxes.append(tb)

        self.tracked_boxes = updated_boxes
        return self.tracked_boxes


# ---------------------------------------------------------------------------
# High-Quality OpenCV Drawing Primitives
# ---------------------------------------------------------------------------

class HUDDrawUtils:
    """Optimized OpenCV rendering utilities for anti-aliased glass panels and shapes."""

    @staticmethod
    def draw_glass_panel(
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        bg_color: tuple[int, int, int] = HUDTheme.PANEL_BG,
        border_color: Optional[tuple[int, int, int]] = HUDTheme.PANEL_BORDER,
        alpha: float = 0.82,
        radius: int = 8,
    ) -> None:
        """Draws a semi-transparent rounded rectangular panel with clean borders."""
        h, w = img.shape[:2]
        x1, y1 = max(0, pt1[0]), max(0, pt1[1])
        x2, y2 = min(w, pt2[0]), min(h, pt2[1])

        if x2 <= x1 or y2 <= y1:
            return

        # Fast alpha blend on ROI
        roi = img[y1:y2, x1:x2]
        panel = np.full_like(roi, bg_color, dtype=np.uint8)
        cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0, roi)

        # Crisp border
        if border_color is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_tech_corners(
        img: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        corner_len: int = 14,
        thickness: int = 2,
    ) -> None:
        """Draws cyberpunk L-shaped corner brackets on a target bounding box."""
        x1, y1 = pt1
        x2, y2 = pt2
        cl = min(corner_len, max(6, int(abs(x2 - x1) * 0.25), int(abs(y2 - y1) * 0.25)))
        t = max(1, thickness)

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, t, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, t, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, t, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, t, cv2.LINE_AA)

    @staticmethod
    def draw_pulsing_glow_border(
        img: np.ndarray,
        color: tuple[int, int, int],
        pulse_phase: float,
        thickness: int = 4,
    ) -> None:
        """Draws a subtle pulsing vignette alert border around the whole viewport."""
        h, w = img.shape[:2]
        intensity = 0.40 + 0.35 * math.sin(pulse_phase)
        glow_color = tuple(int(c * intensity) for c in color)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), glow_color, thickness, cv2.LINE_AA)
        cv2.rectangle(img, (3, 3), (w - 4, h - 4), glow_color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# GuideSenseHUD Main Class
# ---------------------------------------------------------------------------

class GuideSenseHUD:
    """
    State-of-the-Art visual HUD overlay renderer for GuideSense.
    Operates non-blockingly inside the main sensing & decision loop.
    """

    def __init__(
        self,
        window_name: str = "GuideSense — Live Navigation HUD",
        features: Optional[dict[str, str]] = None,
    ) -> None:
        self.window_name = window_name
        self.features: dict[str, str] = features or {
            "Camera": "Active",
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
        self._tracker = DetectionTracker(smoothing_factor=0.65, max_missed=2)
        self._start_time = time.monotonic()

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
        Draws all HUD telemetry, smoothed bounding boxes, spatial guidelines,
        and floating telemetry bars onto a frame copy.
        """
        disp = frame_bgr.copy()
        h, w = disp.shape[:2]
        now = time.monotonic()
        pulse_phase = (now - self._start_time) * 6.0

        # Normalized resolution scaling factor (baseline 640x480)
        scale = max(0.65, min(w / 640.0, h / 480.0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # -------------------------------------------------------------------
        # 1. Urgent Hazard Screen Alert Glow
        # -------------------------------------------------------------------
        if state == SystemState.URGENT:
            HUDDrawUtils.draw_pulsing_glow_border(
                disp, HUDTheme.COLOR_NEAR, pulse_phase, thickness=int(4 * scale)
            )

        # -------------------------------------------------------------------
        # 2. Spatial Lane Guides (LEFT < 33%, CENTER 33-66%, RIGHT > 66%)
        # -------------------------------------------------------------------
        active_directions: set[Direction] = set()
        if result.direction:
            active_directions.add(result.direction)
        for det in frame.detections:
            d = det.direction or compute_direction(det.bbox, w)
            active_directions.add(d)

        if self.show_hud:
            self._draw_spatial_lane_guides(disp, w, h, scale, font, active_directions)

        # -------------------------------------------------------------------
        # 3. Smoothed Bounding Boxes, Reticles & Identification Pills
        # -------------------------------------------------------------------
        tracked_boxes = self._tracker.update(frame.detections, w, h)
        for tb in tracked_boxes:
            self._draw_detection_box(disp, tb, w, h, scale, font)

        # -------------------------------------------------------------------
        # 4. Floating Top Telemetry Header Panel
        # -------------------------------------------------------------------
        self._draw_top_telemetry_bar(disp, frame, result, state, w, scale, font, pulse_phase)

        # -------------------------------------------------------------------
        # 5. Floating Bottom Status & Subsystems Bar
        # -------------------------------------------------------------------
        self._draw_bottom_status_bar(disp, h, w, scale, font)

        # -------------------------------------------------------------------
        # 6. Center Horizon Clearance Crosshair
        # -------------------------------------------------------------------
        if self.show_hud and not tracked_boxes:
            self._draw_horizon_reticle(disp, w, h, scale)

        # -------------------------------------------------------------------
        # 7. Pause Modal Overlay
        # -------------------------------------------------------------------
        if self.is_paused:
            self._draw_pause_overlay(disp, w, h, scale, font)

        return disp

    def _draw_spatial_lane_guides(
        self,
        disp: np.ndarray,
        w: int,
        h: int,
        scale: float,
        font: int,
        active_directions: set[Direction],
    ) -> None:
        """Renders subtle, high-tech vertical zone lines and lane indicator pills."""
        x_left = int(w * 0.33)
        x_right = int(w * 0.66)
        top_y = int(52 * scale)
        bot_y = h - int(40 * scale)

        # Subtle vertical dashed lane dividers
        dash_len = int(12 * scale)
        gap_len = int(8 * scale)
        for x in (x_left, x_right):
            curr_y = top_y
            while curr_y < bot_y:
                next_y = min(curr_y + dash_len, bot_y)
                cv2.line(disp, (x, curr_y), (x, next_y), (50, 58, 70), 1, cv2.LINE_AA)
                curr_y += dash_len + gap_len

        # Subtle bottom lane tags
        lane_font_scale = 0.38 * scale
        tag_y = h - int(48 * scale)

        lanes = [
            ("LEFT LANE", int(w * 0.04), int(w * 0.29), Direction.LEFT),
            ("CENTER PATH", int(w * 0.37), int(w * 0.62), Direction.CENTER),
            ("RIGHT LANE", int(w * 0.70), int(w * 0.96), Direction.RIGHT),
        ]

        for title, lx1, lx2, direct in lanes:
            is_active = direct in active_directions
            text_color = HUDTheme.ACCENT_CYAN if is_active else (110, 120, 135)
            badge_bg = (30, 38, 48) if is_active else (18, 22, 28)
            border_col = (100, 150, 200) if is_active else (40, 48, 58)

            cx = (lx1 + lx2) // 2
            (tw, th), _ = cv2.getTextSize(title, font, lane_font_scale, 1)
            px1, py1 = cx - tw // 2 - 8, tag_y - th - 4
            px2, py2 = cx + tw // 2 + 8, tag_y + 4

            HUDDrawUtils.draw_glass_panel(disp, (px1, py1), (px2, py2), bg_color=badge_bg, border_color=border_col, alpha=0.75)
            cv2.putText(disp, title, (cx - tw // 2, tag_y - 2), font, lane_font_scale, text_color, 1, cv2.LINE_AA)

    def _draw_detection_box(
        self,
        disp: np.ndarray,
        tb: TrackedBox,
        w: int,
        h: int,
        scale: float,
        font: int,
    ) -> None:
        """Draws smoothed target bounding box, corner reticles, and sleek identification pill."""
        x1, y1 = max(0, int(tb.x1)), max(0, int(tb.y1))
        x2, y2 = min(w, int(tb.x2)), min(h, int(tb.y2))

        if x2 <= x1 or y2 <= y1:
            return

        # Pick theme color based on zone
        if tb.zone == Zone.NEAR:
            base_color = HUDTheme.COLOR_NEAR
            box_thickness = max(2, int(3 * scale))
        elif tb.zone == Zone.MID:
            base_color = HUDTheme.COLOR_MID
            box_thickness = max(2, int(2 * scale))
        else:
            base_color = HUDTheme.COLOR_FAR
            box_thickness = max(1, int(1.5 * scale))

        # 1. Subtle semi-transparent bounding box interior tint
        roi = disp[y1:y2, x1:x2]
        if roi.size > 0:
            tint = np.full_like(roi, base_color, dtype=np.uint8)
            fill_alpha = 0.10 * tb.alpha
            cv2.addWeighted(tint, fill_alpha, roi, 1.0 - fill_alpha, 0, roi)

        # 2. Bounding box outer frame
        cv2.rectangle(disp, (x1, y1), (x2, y2), base_color, 1, cv2.LINE_AA)

        # 3. High-tech corner target brackets
        HUDDrawUtils.draw_tech_corners(
            disp,
            (x1, y1),
            (x2, y2),
            base_color,
            corner_len=int(16 * scale),
            thickness=box_thickness,
        )

        if not self.show_hud:
            return

        # 4. Identification Header Pill Badge
        badge_text = (
            f"{tb.label.upper()} {int(tb.confidence * 100)}%  •  "
            f"{tb.distance_m:.2f}m  •  {tb.direction.name}"
        )
        badge_font_scale = 0.42 * scale
        (tw, th), _ = cv2.getTextSize(badge_text, font, badge_font_scale, 1)

        pill_pad_x = int(8 * scale)
        pill_pad_y = int(5 * scale)
        pill_w = tw + pill_pad_x * 2 + int(10 * scale)
        pill_h = th + pill_pad_y * 2

        # Position pill cleanly above bbox if space permits, else inside
        if y1 >= (pill_h + 8):
            by1 = y1 - pill_h - 4
            by2 = y1 - 4
        else:
            by1 = y1 + 4
            by2 = y1 + pill_h + 4

        bx1 = max(0, x1)
        bx2 = min(w, bx1 + pill_w)

        # Draw glass pill background
        HUDDrawUtils.draw_glass_panel(
            disp,
            (bx1, by1),
            (bx2, by2),
            bg_color=HUDTheme.PANEL_BG,
            border_color=base_color,
            alpha=0.88,
        )

        # Color dot indicator on badge
        dot_radius = max(2, int(3.5 * scale))
        dot_cx = bx1 + pill_pad_x + dot_radius
        dot_cy = (by1 + by2) // 2
        cv2.circle(disp, (dot_cx, dot_cy), dot_radius, base_color, -1, cv2.LINE_AA)

        # Text label
        text_x = dot_cx + dot_radius + int(6 * scale)
        text_y = dot_cy + th // 2 - 1
        cv2.putText(
            disp,
            badge_text,
            (text_x, text_y),
            font,
            badge_font_scale,
            HUDTheme.TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

    def _draw_top_telemetry_bar(
        self,
        disp: np.ndarray,
        frame: SensorFrame,
        result: FusionResult,
        state: SystemState,
        w: int,
        scale: float,
        font: int,
        pulse_phase: float,
    ) -> None:
        """Renders floating top glass bar with GuideSense branding, State Badge, and Distance Gauge."""
        bar_h = int(46 * scale)
        HUDDrawUtils.draw_glass_panel(
            disp,
            (0, 0),
            (w, bar_h),
            bg_color=HUDTheme.PANEL_BG,
            border_color=HUDTheme.PANEL_BORDER,
            alpha=0.90,
        )

        # --- Left Section: Brand & Telemetry ---
        dot_color = (
            HUDTheme.STATE_URGENT if state == SystemState.URGENT else (
                HUDTheme.STATE_INFORMATIVE if state == SystemState.INFORMATIVE else HUDTheme.STATE_SILENT
            )
        )
        dot_cx = int(18 * scale)
        dot_cy = int(bar_h / 2)
        dot_r = max(3, int(4.5 * scale))
        # Subtle pulsing dot
        pulsed_r = dot_r + int((math.sin(pulse_phase) + 1.0) * 1.2)
        cv2.circle(disp, (dot_cx, dot_cy), pulsed_r, dot_color, -1, cv2.LINE_AA)

        brand_text = "GUIDESENSE"
        cv2.putText(
            disp,
            brand_text,
            (dot_cx + dot_r + int(8 * scale), dot_cy + int(5 * scale)),
            font,
            0.52 * scale,
            HUDTheme.TEXT_PRIMARY,
            1,
            cv2.LINE_AA,
        )

        fps_str = f"{self._fps:.1f} FPS"
        cv2.putText(
            disp,
            fps_str,
            (dot_cx + dot_r + int(120 * scale), dot_cy + int(4 * scale)),
            font,
            0.38 * scale,
            HUDTheme.TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

        # --- Center Section: System State Badge Pill ---
        if state == SystemState.URGENT:
            state_bg = HUDTheme.STATE_URGENT
            state_text_col = HUDTheme.WHITE
            state_label = "URGENT • HAZARD NEAR"
        elif state == SystemState.INFORMATIVE:
            state_bg = HUDTheme.STATE_INFORMATIVE
            state_text_col = HUDTheme.TEXT_DARK
            lbl_suffix = f" [{str(result.label).upper()}]" if result.label else ""
            state_label = f"INFORMATIVE{lbl_suffix}"
        else:
            state_bg = HUDTheme.STATE_SILENT
            state_text_col = HUDTheme.TEXT_DARK
            state_label = "SILENT • PATH CLEAR"

        (stw, sth), _ = cv2.getTextSize(state_label, font, 0.44 * scale, 1)
        pill_cx = w // 2
        sx1 = pill_cx - stw // 2 - int(12 * scale)
        sx2 = pill_cx + stw // 2 + int(12 * scale)
        sy1 = int(7 * scale)
        sy2 = bar_h - int(7 * scale)

        # Glass pill for state
        cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), state_bg, -1)
        cv2.putText(
            disp,
            state_label,
            (pill_cx - stw // 2, (sy1 + sy2) // 2 + sth // 2),
            font,
            0.44 * scale,
            state_text_col,
            1,
            cv2.LINE_AA,
        )

        # --- Right Section: Metric Distance Gauge ---
        dist_val = frame.distance_m
        if dist_val < NEAR_THRESHOLD_M:
            zone_tag = "NEAR"
            zone_color = HUDTheme.COLOR_NEAR
        elif dist_val <= MID_THRESHOLD_M:
            zone_tag = "MID"
            zone_color = HUDTheme.COLOR_MID
        else:
            zone_tag = "FAR"
            zone_color = HUDTheme.COLOR_FAR

        dist_label = f"{dist_val:.2f}m [{zone_tag}]"
        (dtw, dth), _ = cv2.getTextSize(dist_label, font, 0.46 * scale, 1)
        dist_x = w - dtw - int(16 * scale)
        dist_y = dot_cy + int(5 * scale)

        cv2.putText(
            disp,
            dist_label,
            (dist_x, dist_y),
            font,
            0.46 * scale,
            zone_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_bottom_status_bar(
        self,
        disp: np.ndarray,
        h: int,
        w: int,
        scale: float,
        font: int,
    ) -> None:
        """Renders bottom telemetry footer with active hardware chips and keyboard hints."""
        bot_h = int(32 * scale)
        bot_y1 = h - bot_h
        HUDDrawUtils.draw_glass_panel(
            disp,
            (0, bot_y1),
            (w, h),
            bg_color=HUDTheme.PANEL_BG,
            border_color=HUDTheme.PANEL_BORDER,
            alpha=0.90,
        )

        # Active feature chips
        items = [
            ("CAM", self.features.get("Camera", "Active")),
            ("GEMINI", self.features.get("Gemini", "Off")),
            ("BACKBOARD", self.features.get("Backboard", "Off")),
            ("LOG", self.features.get("Logging", "Active")),
        ]

        curr_x = int(12 * scale)
        text_y = h - int(11 * scale)

        for name, val in items:
            is_on = val not in ("Off", "Mock", "None", "Disabled")
            status_col = HUDTheme.STATE_SILENT if is_on else HUDTheme.TEXT_MUTED
            chip_text = f"{name}:{val}"
            (tw, _), _ = cv2.getTextSize(chip_text, font, 0.36 * scale, 1)

            cv2.putText(disp, chip_text, (curr_x, text_y), font, 0.36 * scale, status_col, 1, cv2.LINE_AA)
            curr_x += tw + int(14 * scale)

        # Right keybind hints
        controls = "[Q] Quit  •  [G] HUD  •  [SPACE] Pause"
        (ctw, _), _ = cv2.getTextSize(controls, font, 0.36 * scale, 1)
        cv2.putText(
            disp,
            controls,
            (w - ctw - int(12 * scale), text_y),
            font,
            0.36 * scale,
            HUDTheme.TEXT_SECONDARY,
            1,
            cv2.LINE_AA,
        )

    def _draw_horizon_reticle(self, disp: np.ndarray, w: int, h: int, scale: float) -> None:
        """Draws a subtle assistive center crosshair when no obstacles are detected."""
        cx, cy = w // 2, h // 2
        r = int(16 * scale)
        col = (70, 80, 95)

        cv2.circle(disp, (cx, cy), r, col, 1, cv2.LINE_AA)
        cv2.line(disp, (cx - r - 6, cy), (cx - 4, cy), col, 1, cv2.LINE_AA)
        cv2.line(disp, (cx + 4, cy), (cx + r + 6, cy), col, 1, cv2.LINE_AA)
        cv2.line(disp, (cx, cy - r - 6), (cx, cy - 4), col, 1, cv2.LINE_AA)
        cv2.line(disp, (cx, cy + 4), (cx, cy + r + 6), col, 1, cv2.LINE_AA)

    def _draw_pause_overlay(
        self,
        disp: np.ndarray,
        w: int,
        h: int,
        scale: float,
        font: int,
    ) -> None:
        """Draws frosted glass pause modal in the center of the screen."""
        pause_text = "STREAM PAUSED"
        sub_text = "PRESS [SPACE] TO RESUME"
        (ptw, pth), _ = cv2.getTextSize(pause_text, font, 0.72 * scale, 2)
        (stw, sth), _ = cv2.getTextSize(sub_text, font, 0.44 * scale, 1)

        card_w = max(ptw, stw) + int(50 * scale)
        card_h = pth + sth + int(50 * scale)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2

        HUDDrawUtils.draw_glass_panel(
            disp,
            (x1, y1),
            (x2, y2),
            bg_color=HUDTheme.PANEL_BG,
            border_color=HUDTheme.COLOR_MID,
            alpha=0.92,
        )

        cv2.putText(
            disp,
            pause_text,
            (cx - ptw // 2, y1 + int(32 * scale)),
            font,
            0.72 * scale,
            HUDTheme.WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            sub_text,
            (cx - stw // 2, y1 + int(32 * scale) + sth + int(16 * scale)),
            font,
            0.44 * scale,
            HUDTheme.COLOR_MID,
            1,
            cv2.LINE_AA,
        )

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
        """Generates a futuristic synthetic digital canvas for mock mode."""
        canvas = np.full((height, width, 3), 22, dtype=np.uint8)

        # Subtle dark digital grid
        grid_step = 40
        for y in range(0, height, grid_step):
            cv2.line(canvas, (0, y), (width, y), HUDTheme.GRID_LINE, 1)
        for x in range(0, width, grid_step):
            cv2.line(canvas, (x, 0), (x, height), HUDTheme.GRID_LINE, 1)

        cv2.putText(
            canvas,
            "SIMULATED CAMERA STREAM",
            (int(width * 0.28), int(height * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            HUDTheme.TEXT_MUTED,
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
