"""
vision/calibration.py - Adaptive hand scale, resting center, and active workspace calibration.
Allows users to calibrate tracking for their unique hand size, seating distance, and comfortable arm reach.
Includes dynamic motion scale amplification so natural hand movements reach all desktop corners without leaving the camera frame.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from vision.landmarks import get_hand_scale, get_palm_center, LandmarkPoint


class HandCalibrator:
    """
    Manages active interaction workspace mapping and hand scale calibration.
    """
    def __init__(self,
                 margin_x: float = 0.08,
                 margin_y: float = 0.08,
                 motion_scale: float = 1.20,
                 sampling_frames: int = 30):
        """
        :param margin_x: Normalized margin left/right (default 0.08 = 84% active frame width).
        :param margin_y: Normalized margin top/bottom (default 0.08 = 84% active frame height).
        :param motion_scale: Reach amplification multiplier (default 1.20x) balanced for steady pointing and screen-wide reach.
        :param sampling_frames: Number of frames to average during calibration routine.
        """
        self.default_margin_x = margin_x
        self.default_margin_y = margin_y
        self.motion_scale = motion_scale
        self.sampling_frames = sampling_frames

        # Active workspace bounds in normalized [0.0, 1.0] coordinates
        self.min_x = margin_x
        self.max_x = 1.0 - margin_x
        self.min_y = margin_y
        self.max_y = 1.0 - margin_y

        # Baseline resting hand metrics
        self.baseline_scale: float = 0.20
        self.resting_center: Tuple[float, float] = (0.5, 0.5)

        # Calibration state machine
        self.is_calibrating: bool = False
        self.calibration_progress: float = 0.0
        self._samples_scale: List[float] = []
        self._samples_center: List[Tuple[float, float]] = []

    def adjust_scale(self, delta: float) -> float:
        """Increases or decreases movement amplification multiplier."""
        self.motion_scale = max(0.8, min(3.0, self.motion_scale + delta))
        print(f"[Calibration] Motion Scale set to: {self.motion_scale:.2f}x")
        return self.motion_scale

    def adjust_margins(self, delta: float):
        """Expands (delta < 0) or contracts (delta > 0) active workspace borders."""
        new_mx = max(0.02, min(0.35, self.default_margin_x + delta))
        new_my = max(0.02, min(0.35, self.default_margin_y + delta))
        self.default_margin_x = new_mx
        self.default_margin_y = new_my
        self.min_x = new_mx
        self.max_x = 1.0 - new_mx
        self.min_y = new_my
        self.max_y = 1.0 - new_my
        print(f"[Calibration] Workspace active span: {int((1.0 - 2 * new_mx) * 100)}% width")

    def start_calibration(self):
        """Begins calibration routine over the next N frames."""
        self.is_calibrating = True
        self.calibration_progress = 0.0
        self._samples_scale.clear()
        self._samples_center.clear()
        print("[Calibration] Started hand calibration. Hold your hand in resting position...")

    def update_calibration(self, landmarks: List[LandmarkPoint]) -> bool:
        """
        Feed landmarks during active calibration.
        Returns True when calibration completes.
        """
        if not self.is_calibrating or not landmarks:
            return False

        scale = get_hand_scale(landmarks)
        palm = get_palm_center(landmarks)

        self._samples_scale.append(scale)
        self._samples_center.append((palm.x, palm.y))

        count = len(self._samples_scale)
        self.calibration_progress = min(1.0, count / float(self.sampling_frames))

        if count >= self.sampling_frames:
            avg_scale = sum(self._samples_scale) / count
            avg_x = sum(p[0] for p in self._samples_center) / count
            avg_y = sum(p[1] for p in self._samples_center) / count

            self.baseline_scale = max(0.05, avg_scale)
            self.resting_center = (avg_x, avg_y)

            # Center active workspace around the user's natural arm position
            span_x = (1.0 - 2 * self.default_margin_x) / 2.0
            span_y = (1.0 - 2 * self.default_margin_y) / 2.0

            self.min_x = max(0.02, min(0.40, avg_x - span_x))
            self.max_x = min(0.98, max(0.60, avg_x + span_x))
            self.min_y = max(0.02, min(0.40, avg_y - span_y))
            self.max_y = min(0.98, max(0.60, avg_y + span_y))

            self.is_calibrating = False
            print(f"[Calibration] Completed! Baseline Scale={self.baseline_scale:.3f}, "
                  f"Resting Center=({avg_x:.2f}, {avg_y:.2f}), "
                  f"Workspace=[X: {self.min_x:.2f}-{self.max_x:.2f}, Y: {self.min_y:.2f}-{self.max_y:.2f}]")
            return True

        return False

    def reset_defaults(self):
        """Resets workspace and scale to factory defaults."""
        self.default_margin_x = 0.08
        self.default_margin_y = 0.08
        self.min_x = self.default_margin_x
        self.max_x = 1.0 - self.default_margin_x
        self.min_y = self.default_margin_y
        self.max_y = 1.0 - self.default_margin_y
        self.motion_scale = 1.20
        self.baseline_scale = 0.20
        self.resting_center = (0.5, 0.5)
        self.is_calibrating = False

    def map_camera_to_screen(self,
                             cam_norm_x: float,
                             cam_norm_y: float,
                             screen_w: int,
                             screen_h: int) -> Tuple[int, int]:
        """
        Maps normalized camera coordinates [0.0, 1.0] to desktop screen resolution [0, screen_w], [0, screen_h].
        Uses motion scale amplification centered on the active workspace so comfortable hand movements
        reach all corners of the monitor without pushing the hand outside the physical camera lens.
        """
        # Center of active interaction area
        center_x = (self.min_x + self.max_x) / 2.0
        center_y = (self.min_y + self.max_y) / 2.0
        half_span_x = max(0.01, (self.max_x - self.min_x) / 2.0)
        half_span_y = max(0.01, (self.max_y - self.min_y) / 2.0)

        # Relative offset from center normalized in [-1.0, +1.0]
        rel_x = (cam_norm_x - center_x) / half_span_x
        rel_y = (cam_norm_y - center_y) / half_span_y

        # Apply motion scale amplification factor
        scaled_rel_x = rel_x * self.motion_scale
        scaled_rel_y = rel_y * self.motion_scale

        # Map to monitor coordinates [0.0, 1.0]
        norm_scr_x = 0.5 + (scaled_rel_x * 0.5)
        norm_scr_y = 0.5 + (scaled_rel_y * 0.5)

        # Smooth boundary clamping
        norm_scr_x = max(0.0, min(1.0, norm_scr_x))
        norm_scr_y = max(0.0, min(1.0, norm_scr_y))

        screen_x = int(norm_scr_x * screen_w)
        screen_y = int(norm_scr_y * screen_h)

        return screen_x, screen_y

    def get_workspace_rect(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Returns (x1, y1, x2, y2) pixel bounding box in camera coordinates for HUD visualization.
        """
        x1 = int(self.min_x * width)
        y1 = int(self.min_y * height)
        x2 = int(self.max_x * width)
        y2 = int(self.max_y * height)
        return x1, y1, x2, y2
