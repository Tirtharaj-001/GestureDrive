"""
vr/desktop_gestures.py - VRChat Desktop Mode Native Gesture Keyboard Emulation.
Simulates VRChat desktop gesture hotkeys (Left Shift / Right Shift + F1..F8)
so that avatars physically animate their hands and arms in Desktop (Non-VR) Mode,
even on default starter avatars that lack custom OSC Avatar 3.0 parameters.
"""

import sys
import time
from typing import Dict, Optional

# Windows Virtual Key Codes
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_F1 = 0x70  # Neutral / Idle
VK_F2 = 0x71  # Fist / Grab
VK_F3 = 0x72  # Hand Open
VK_F4 = 0x73  # Point
VK_F5 = 0x74  # Victory / Peace
VK_F6 = 0x75  # Rock and Roll
VK_F7 = 0x76  # Hand Gun
VK_F8 = 0x77  # Thumbs Up

# Desktop Locomotion Keys (WASD + Left/Right Arrow Turn)
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_LEFT = 0x25
VK_RIGHT = 0x27

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
MAPVK_VK_TO_VSC = 0

# Comprehensive gesture mapping to VRChat function keys
GESTURE_TO_FKEY = {
    "Fist": VK_F2,
    "Pointing": VK_F4,
    "Peace": VK_F5,
    "Thumbs Up": VK_F8,
    "Rock On": VK_F6,
    "Hand Gun": VK_F7,
    "Pinch": VK_F2,     # Pinch acts as Fist / Grab in VRChat
    "Open Palm": VK_F3, # Open hand
    "None": None,       # Release
}


class VRChatDesktopGestureEmulator:
    """
    Emulates VRChat Desktop keyboard hotkeys (Shift + F1..F8)
    and WASD/Arrow walking with hardware scan codes for DirectInput/Unity compatibility.
    Left hand uses Left Shift; Right hand uses Right Shift.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.is_windows = sys.platform == "win32"
        self._user32 = None
        if self.is_windows:
            try:
                import ctypes
                self._user32 = ctypes.windll.user32
            except Exception:
                self._user32 = None

        # Track currently held keys per hand: "Left": fkey_vk or None, "Right": fkey_vk or None
        self._current_keys: Dict[str, Optional[int]] = {
            "Left": None,
            "Right": None,
        }
        # Track currently held movement keys (WASD, Arrows)
        self._held_movement_keys = set()

    def _send_key(self, vk: int, down: bool):
        """Dispatches key event with hardware scan code for DirectX/Unity compatibility."""
        if not self._user32:
            return
        scan = self._user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
        flags = 0
        if not down:
            flags |= KEYEVENTF_KEYUP
        if vk in (VK_RSHIFT, VK_LEFT, VK_RIGHT):
            flags |= KEYEVENTF_EXTENDEDKEY
        self._user32.keybd_event(vk, scan, flags, 0)

    def is_vrchat_foreground(self) -> bool:
        """Returns True if VRChat is currently the focused foreground window."""
        if not self._user32:
            return False
        try:
            import ctypes
            hwnd = self._user32.GetForegroundWindow()
            if not hwnd:
                return False
            buf = ctypes.create_unicode_buffer(256)
            self._user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value.lower()
            return "vrchat" in title
        except Exception:
            return True

    def update_locomotion(self, joy_x: float, joy_y: float):
        """
        Drives VRChat desktop walking using W, A, S, D based on Left Hand tilt.
        joy_y > 0.20 -> W (Forward)
        joy_y < -0.20 -> S (Backward)
        joy_x < -0.20 -> A (Strafe Left)
        joy_x > 0.20 -> D (Strafe Right)
        """
        if not self.enabled or not self._user32 or not self.is_vrchat_foreground():
            self.release_movement()
            return

        needed = set()
        if joy_y > 0.20:
            needed.add(VK_W)
        elif joy_y < -0.20:
            needed.add(VK_S)

        if joy_x < -0.20:
            needed.add(VK_A)
        elif joy_x > 0.20:
            needed.add(VK_D)

        to_release = (self._held_movement_keys & {VK_W, VK_A, VK_S, VK_D}) - needed
        for vk in list(to_release):
            self._send_key(vk, False)
            self._held_movement_keys.remove(vk)

        to_press = needed - self._held_movement_keys
        for vk in list(to_press):
            self._send_key(vk, True)
            self._held_movement_keys.add(vk)

    def update_look(self, look_x: float):
        """
        Drives VRChat desktop turning using Left/Right Arrow keys based on Right Hand tilt.
        look_x < -0.20 -> Left Arrow (Turn Left)
        look_x > 0.20 -> Right Arrow (Turn Right)
        """
        if not self.enabled or not self._user32 or not self.is_vrchat_foreground():
            self.release_look()
            return

        needed = set()
        if look_x < -0.20:
            needed.add(VK_LEFT)
        elif look_x > 0.20:
            needed.add(VK_RIGHT)

        to_release = (self._held_movement_keys & {VK_LEFT, VK_RIGHT}) - needed
        for vk in list(to_release):
            self._send_key(vk, False)
            self._held_movement_keys.remove(vk)

        to_press = needed - self._held_movement_keys
        for vk in list(to_press):
            self._send_key(vk, True)
            self._held_movement_keys.add(vk)

    def release_movement(self):
        """Releases WASD movement keys."""
        if not self._user32:
            return
        wasd = {VK_W, VK_A, VK_S, VK_D} & self._held_movement_keys
        for vk in list(wasd):
            self._send_key(vk, False)
            self._held_movement_keys.remove(vk)

    def release_look(self):
        """Releases Left/Right Arrow keys."""
        if not self._user32:
            return
        arrows = {VK_LEFT, VK_RIGHT} & self._held_movement_keys
        for vk in list(arrows):
            self._send_key(vk, False)
            self._held_movement_keys.remove(vk)

    def update_hand_gesture(self, handedness: str, gesture_name: str, force_send: bool = False):
        """
        Sends native VRChat desktop key combinations with hardware scan codes.
        Left hand uses Left Shift + F1..F8; Right hand uses Right Shift + F1..F8.
        """
        if not self.enabled or not self._user32:
            return

        side = "Left" if str(handedness).lower() == "left" else "Right"
        shift_vk = VK_LSHIFT if side == "Left" else VK_RSHIFT
        target_fkey = GESTURE_TO_FKEY.get(gesture_name, None)

        current_fkey = self._current_keys[side]

        # If gesture hasn't changed, keep holding
        if target_fkey == current_fkey:
            return

        # Only send keystrokes if VRChat is focused (or forced) to prevent typing in other windows
        if not force_send and not self.is_vrchat_foreground():
            return

        # Release previous gesture if one was held
        if current_fkey is not None:
            self._send_key(current_fkey, False)
            self._send_key(shift_vk, False)
            self._current_keys[side] = None

        # Press new gesture if applicable
        if target_fkey is not None:
            self._send_key(shift_vk, True)
            self._send_key(target_fkey, True)
            self._current_keys[side] = target_fkey

    def release_all(self):
        """Releases all held gesture hotkeys and movement keys."""
        if not self._user32:
            return
        for side, fkey in self._current_keys.items():
            if fkey is not None:
                shift_vk = VK_LSHIFT if side == "Left" else VK_RSHIFT
                self._send_key(fkey, False)
                self._send_key(shift_vk, False)
                self._current_keys[side] = None
        self.release_movement()
        self.release_look()
