"""
Translates recognized hand gestures and smoothed coordinates into real-time Windows mouse actions:
- Pointing: Cursor navigation
- Pinch: Left Click / Click-and-Drag
- Peace Sign (2 fingers): Right Click
- Open Palm / Fist: Clutch / Pause movement
Supports ultra-low-latency native Windows User32 API with pyautogui/pynput fallback.
"""

import sys
import time
import math
from typing import Optional, Tuple
from enum import Enum

# Windows native user32 event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800


class MouseActionState(Enum):
    IDLE = "Idle"
    NAVIGATING = "Moving"
    DRAGGING = "Dragging"
    CLICKING = "Left Click"
    RIGHT_CLICK = "Right Click"
    PAUSED = "Paused / Clutch"


class MouseController:
    """
    High-performance OS mouse controller optimized for Windows.
    Uses ctypes.windll.user32 for sub-millisecond dispatch without blocking the main OpenCV thread.
    """
    def __init__(self,
                 click_debounce_sec: float = 0.28,
                 drag_hold_threshold_sec: float = 0.22):
        self.click_debounce_sec = click_debounce_sec
        self.drag_hold_threshold_sec = drag_hold_threshold_sec

        self.enabled = True
        self.state = MouseActionState.IDLE

        # Hardware display resolution
        self.screen_width, self.screen_height = self._get_screen_resolution()

        # State tracking
        self.is_left_down = False
        self.pinch_start_time: Optional[float] = None
        self.last_click_time = 0.0
        self.last_right_click_time = 0.0

        # Position tracking
        self.current_cursor_pos: Tuple[int, int] = (self.screen_width // 2, self.screen_height // 2)

        # Click stabilization (Pinch Anchor) to prevent cursor slipping during clicks
        self.pinch_anchor_pos: Optional[Tuple[int, int]] = None
        self.is_drag_unlocked: bool = False
        self.drag_unlock_distance: float = 16.0  # pixels of movement required to break click lock into drag

        # Initialize OS driver
        self.has_win32 = False
        self._init_os_driver()

    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Detects primary monitor pixel dimensions."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Enable DPI awareness to get true physical resolution
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass

        try:
            import pyautogui
            w, h = pyautogui.size()
            return w, h
        except Exception:
            pass

        return 1920, 1080

    def _init_os_driver(self):
        """Initializes direct ctypes Win32 mouse driver."""
        if sys.platform == "win32":
            try:
                import ctypes
                self.user32 = ctypes.windll.user32
                self.has_win32 = True
                print(f"[MouseController] Initialized native Windows User32 driver ({self.screen_width}x{self.screen_height})")
                return
            except Exception as e:
                print(f"[MouseController] Warning: Could not initialize ctypes Win32 driver: {e}")

        # Fallback to pyautogui
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0
            self.has_win32 = False
            print("[MouseController] Initialized PyAutoGUI fallback driver.")
        except Exception as e:
            print(f"[MouseController] Error: PyAutoGUI driver failed: {e}")

    def move_to(self, x: int, y: int):
        """
        Moves system mouse cursor to target screen coordinate (x, y).
        Dispatches both MOUSEEVENTF_MOVE delta and SetCursorPos so both
        standard desktop apps and 3D pointer-locked games (like Slow Roads) receive motion.
        """
        if not self.enabled:
            return

        clamped_x = max(0, min(self.screen_width - 1, int(x)))
        clamped_y = max(0, min(self.screen_height - 1, int(y)))
        prev_x, prev_y = self.current_cursor_pos
        dx = clamped_x - prev_x
        dy = clamped_y - prev_y
        self.current_cursor_pos = (clamped_x, clamped_y)

        if self.has_win32:
            if dx != 0 or dy != 0:
                # Deliver relative hardware delta for pointer-locked 3D games (Slow Roads / WebGL / FPS)
                self.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
            self.user32.SetCursorPos(clamped_x, clamped_y)
        else:
            try:
                import pyautogui
                pyautogui.moveTo(clamped_x, clamped_y)
            except Exception:
                pass

    def left_down(self):
        """Presses and holds the left mouse button."""
        if not self.enabled or self.is_left_down:
            return
        self.is_left_down = True
        self.state = MouseActionState.DRAGGING
        if self.has_win32:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.mouseDown(button='left')
            except Exception:
                pass

    def left_up(self):
        """Releases the left mouse button."""
        if not self.is_left_down:
            return
        self.is_left_down = False
        self.state = MouseActionState.IDLE
        if self.has_win32:
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.mouseUp(button='left')
            except Exception:
                pass

    def click_left(self):
        """Dispatches an instant left click."""
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_click_time < self.click_debounce_sec:
            return
        self.last_click_time = now
        self.state = MouseActionState.CLICKING
        if self.has_win32:
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.click(button='left')
            except Exception:
                pass

    def click_right(self):
        """Dispatches a right click."""
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_right_click_time < 0.45:
            return
        self.last_right_click_time = now
        self.state = MouseActionState.RIGHT_CLICK
        if self.has_win32:
            self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN | MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        else:
            try:
                import pyautogui
                pyautogui.click(button='right')
            except Exception:
                pass

    def scroll(self, steps: int):
        """Scrolls vertically (positive = scroll up, negative = scroll down)."""
        if not self.enabled:
            return
        if self.has_win32:
            # 120 units per notch in Windows
            self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(steps * 120), 0)
        else:
            try:
                import pyautogui
                pyautogui.scroll(steps * 100)
            except Exception:
                pass

    def toggle_enabled(self) -> bool:
        """Toggles clutch/pause mode on or off."""
        self.enabled = not self.enabled
        if not self.enabled and self.is_left_down:
            self.left_up()
        self.pinch_anchor_pos = None
        self.is_drag_unlocked = False
        self.state = MouseActionState.IDLE if self.enabled else MouseActionState.PAUSED
        print(f"[MouseController] Mouse control {'ENABLED' if self.enabled else 'PAUSED (Clutch)'}")
        return self.enabled

    def update(self,
               gesture_name: str,
               screen_target_pos: Tuple[int, int],
               hand_visible: bool = True):
        """
        State machine that updates mouse navigation and clicks based on continuous gesture inputs.
        Includes Click Stabilization (Pinch Anchor): locks cursor at pinch onset to prevent fingertip drift on click.
        """
        now = time.time()

        if not self.enabled or not hand_visible:
            if self.is_left_down:
                self.left_up()
            self.pinch_start_time = None
            self.pinch_anchor_pos = None
            self.is_drag_unlocked = False
            self.state = MouseActionState.PAUSED if not self.enabled else MouseActionState.IDLE
            return

        tx, ty = screen_target_pos

        # Handle Pinch or OK (Left Click & Drag)
        if gesture_name in ("Pinch", "OK"):
            if self.pinch_start_time is None:
                self.pinch_start_time = now
                self.pinch_anchor_pos = (tx, ty)
                self.is_drag_unlocked = False

            elapsed = now - self.pinch_start_time

            # Check if hand intentionally moved away from anchor to initiate drag
            if self.pinch_anchor_pos is not None and not self.is_drag_unlocked:
                drag_dist = math.hypot(tx - self.pinch_anchor_pos[0], ty - self.pinch_anchor_pos[1])
                if drag_dist > self.drag_unlock_distance:
                    self.is_drag_unlocked = True

            # If drag has been unlocked or held past drag threshold, activate left_down for dragging
            if (self.is_drag_unlocked or elapsed >= self.drag_hold_threshold_sec) and not self.is_left_down:
                self.left_down()

            # Position update: lock cursor rock-solid to pinch anchor until intentional drag begins
            if self.is_drag_unlocked:
                self.move_to(tx, ty)
            elif self.pinch_anchor_pos is not None:
                self.move_to(self.pinch_anchor_pos[0], self.pinch_anchor_pos[1])
            else:
                self.move_to(tx, ty)

            self.state = MouseActionState.DRAGGING if self.is_left_down else MouseActionState.CLICKING

        else:
            # Pinch released
            if self.pinch_start_time is not None:
                elapsed = now - self.pinch_start_time
                if elapsed < self.drag_hold_threshold_sec and not self.is_drag_unlocked:
                    # Quick pinch release = Single Left Click at the stabilized anchor position
                    if self.pinch_anchor_pos is not None:
                        self.move_to(self.pinch_anchor_pos[0], self.pinch_anchor_pos[1])
                    self.click_left()
                elif self.is_left_down:
                    self.left_up()
                self.pinch_start_time = None
                self.pinch_anchor_pos = None
                self.is_drag_unlocked = False

            # Handle Right Click (Peace / Victory sign)
            if gesture_name == "Peace":
                self.click_right()
                self.state = MouseActionState.RIGHT_CLICK

            # Handle Cursor Navigation (Pointing)
            elif gesture_name == "Pointing":
                self.move_to(tx, ty)
                self.state = MouseActionState.NAVIGATING

            # Clutch / Pause Hand Poses (Open Palm / Fist)
            elif gesture_name in ("Open Palm", "Fist"):
                self.state = MouseActionState.PAUSED
                # Cursor position holds still

            else:
                self.state = MouseActionState.IDLE
