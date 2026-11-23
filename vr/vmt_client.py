"""
vr/vmt_client.py - Day 5: VR Controller Buttons, Triggers & Thumbsticks Bridge.
Spawns and streams real-time 6-DOF virtual controllers or Vive Trackers into SteamVR
over UDP/OSC at 60 Hz using the Virtual Motion Tracker (VMT) OpenVR driver protocol.

Protocol Reference (VMT by gpsnmeajp):
  Port: 39570 (UDP)
  /VMT/Room/Unity  (int)idx, (int)enable, (float)timeoffset, (float)x, (float)y, (float)z, (float)qx, (float)qy, (float)qz, (float)qw
  /VMT/Room/UEuler (int)idx, (int)enable, (float)timeoffset, (float)x, (float)y, (float)z, (float)rx, (float)ry, (float)rz
  /VMT/Input/Trigger (int)idx, (int)triggerIdx, (float)timeoffset, (float)value
  /VMT/Input/Button  (int)idx, (int)buttonIdx, (float)timeoffset, (int)value
  /VMT/Input/Joystick (int)idx, (int)joystickIdx, (float)timeoffset, (float)x, (float)y
  /VMT/Input/Joystick/Touch (int)idx, (int)joystickIdx, (float)timeoffset, (int)value
  /VMT/Input/Joystick/Click (int)idx, (int)joystickIdx, (float)timeoffset, (int)value
  /VMT/Reset

Day 5 Inputs:
  - Analog Trigger (0.0 -> 1.0): Continuous pinch distance (Index Tip to Thumb Tip).
  - Grip Button: Closed Fist & 3-finger flexion (grabs objects, handles, weapons).
  - A/B / X/Y Buttons: Thumbs Up (B/Y) or Two-finger tap / Peace (A/X).
  - Locomotion / Thumbstick: Pointing hand tilt angle (pitch & roll) mapped to Left Thumbstick.
"""

import math
import time
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any, Union

from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder

from vision.landmarks import (
    WRIST,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    dist_3d,
)


class VMTDeviceRole(IntEnum):
    """VMT device registration roles."""
    DISABLED = 0
    TRACKER_GENERIC = 1
    CONTROLLER_LEFT = 2
    CONTROLLER_RIGHT = 3
    TRACKING_REFERENCE = 4
    CONTROLLER_INDEX_LEFT = 5
    CONTROLLER_INDEX_RIGHT = 6
    TRACKER_VIVE = 7


class VMTMode(IntEnum):
    """User-selectable emulation mode."""
    CONTROLLERS = 1       # Spawns Left (2) & Right (3) VR Hand Controllers
    TRACKERS = 2          # Spawns Left (7) & Right (7) Vive Trackers on Wrists
    INDEX_CONTROLLERS = 3 # Spawns Left (5) & Right (6) Valve Index Knuckles


class _VMTHandState:
    """Maintains tracking cache, rate limiting, and button states for one hand."""
    def __init__(self, index: int, default_role: VMTDeviceRole):
        self.index = index
        self.role = default_role
        self.is_spawned = False
        self.last_send_time = 0.0
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_euler = (0.0, 0.0, 0.0)
        self.last_trigger = -1.0
        self.last_trigger_click = -1
        self.last_grip = -1.0
        self.last_grip_button = -1
        self.last_button_a = -1
        self.last_button_b = -1
        self.last_joy_x = 0.0
        self.last_joy_y = 0.0
        self.last_joy_touch = -1

    def reset(self):
        self.is_spawned = False
        self.last_send_time = 0.0
        self.last_pos = (0.0, 0.0, 0.0)
        self.last_euler = (0.0, 0.0, 0.0)
        self.last_trigger = -1.0
        self.last_trigger_click = -1
        self.last_grip = -1.0
        self.last_grip_button = -1
        self.last_button_a = -1
        self.last_button_b = -1
        self.last_joy_x = 0.0
        self.last_joy_y = 0.0
        self.last_joy_touch = -1


class VMTClient:
    """
    Client for the OpenVR Virtual Motion Tracker (VMT) driver.
    Broadcasts real-time 6-DOF transforms, analog triggers, grips, buttons,
    and pointing tilt locomotion thumbsticks to SteamVR at 60 Hz.
    """
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 39570,
        mode: VMTMode = VMTMode.CONTROLLERS,
        target_hz: float = 60.0,
        broadcast_euler: bool = True,
        broadcast_quaternion: bool = True,
        locomotion_deadzone_deg: float = 8.0,
        locomotion_max_tilt_deg: float = 35.0,
    ):
        self.host = host
        self.port = int(port)
        self.mode = mode
        self.min_interval_sec = 1.0 / max(1.0, target_hz)
        self.broadcast_euler = broadcast_euler
        self.broadcast_quaternion = broadcast_quaternion
        self.locomotion_deadzone_deg = locomotion_deadzone_deg
        self.locomotion_max_tilt_deg = locomotion_max_tilt_deg

        self.enabled = True
        self.client = udp_client.SimpleUDPClient(self.host, self.port)

        # Standard device indices: 0 = Left Hand, 1 = Right Hand
        self.hands: Dict[str, _VMTHandState] = {
            "Left": _VMTHandState(index=0, default_role=self._get_role("Left")),
            "Right": _VMTHandState(index=1, default_role=self._get_role("Right")),
        }

        self.packets_sent: int = 0
        self.last_error: Optional[str] = None
        self.last_summary: Dict[str, Dict[str, Any]] = {}

    def _get_role(self, side: str) -> VMTDeviceRole:
        """Determines the VMT device enable code based on active mode and hand side."""
        if self.mode == VMTMode.CONTROLLERS:
            return VMTDeviceRole.CONTROLLER_LEFT if side == "Left" else VMTDeviceRole.CONTROLLER_RIGHT
        elif self.mode == VMTMode.INDEX_CONTROLLERS:
            return VMTDeviceRole.CONTROLLER_INDEX_LEFT if side == "Left" else VMTDeviceRole.CONTROLLER_INDEX_RIGHT
        else:
            return VMTDeviceRole.TRACKER_VIVE

    def set_mode(self, new_mode: VMTMode):
        """Switches device mode (Controllers <-> Trackers) and updates hand roles."""
        if self.mode != new_mode:
            self.despawn_devices()
            self.mode = new_mode
            for side, state in self.hands.items():
                state.role = self._get_role(side)
            if self.enabled:
                self.spawn_devices()
            print(f"[VMT] Mode switched to: {self.mode.name}")

    def toggle_mode(self) -> VMTMode:
        """Cycles through available VMT emulation modes."""
        modes = [VMTMode.CONTROLLERS, VMTMode.TRACKERS, VMTMode.INDEX_CONTROLLERS]
        next_idx = (modes.index(self.mode) + 1) % len(modes)
        self.set_mode(modes[next_idx])
        return self.mode

    def toggle_enabled(self) -> bool:
        """Toggles SteamVR VMT broadcasting ON or OFF."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.despawn_devices()
        else:
            self.spawn_devices()
        print(f"[VMT] Streaming {'ENABLED' if self.enabled else 'PAUSED'} -> {self.host}:{self.port}")
        return self.enabled

    def spawn_devices(self):
        """
        Broadcasts registration packets for both hands so virtual controller / tracker
        models immediately manifest in SteamVR Home and the SteamVR status overlay.
        """
        for side, state in self.hands.items():
            state.role = self._get_role(side)
            state.is_spawned = True
            px, py, pz = (-0.25 if side == "Left" else +0.25, -0.20, 0.40)
            self._send_raw_room_pose(
                index=state.index,
                enable=int(state.role),
                pos=(px, py, pz),
                quat=(0.0, 0.0, 0.0, 1.0),
                euler=(0.0, 0.0, 0.0),
            )
        print(f"[VMT] Spawned virtual SteamVR devices ({self.mode.name}) on indices [0=Left, 1=Right]")

    def despawn_devices(self):
        """Sends enable=0 and neutral inputs to SteamVR to power off and hide the virtual controllers."""
        for side, state in self.hands.items():
            self.send_input_trigger(state.index, trigger_idx=0, value=0.0)
            self.send_input_trigger(state.index, trigger_idx=1, value=0.0)
            self.send_input_button(state.index, button_idx=0, value=0)
            self.send_input_button(state.index, button_idx=1, value=0)
            self.send_input_button(state.index, button_idx=2, value=0)
            self.send_input_button(state.index, button_idx=3, value=0)
            self.send_input_joystick(state.index, joystick_idx=0, x=0.0, y=0.0)
            self.send_input_joystick(state.index, joystick_idx=1, x=0.0, y=0.0)
            self.send_input_joystick_touch(state.index, joystick_idx=0, value=0)
            self.send_input_joystick_touch(state.index, joystick_idx=1, value=0)

            self._send_raw_room_pose(
                index=state.index,
                enable=int(VMTDeviceRole.DISABLED),
                pos=(0.0, 0.0, 0.0),
                quat=(0.0, 0.0, 0.0, 1.0),
                euler=(0.0, 0.0, 0.0),
            )
            state.reset()
        self.last_summary.clear()
        print("[VMT] Despawned all virtual controllers in SteamVR (enable=0)")

    def _send_raw_room_pose(
        self,
        index: int,
        enable: int,
        pos: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
        euler: Tuple[float, float, float],
    ) -> bool:
        """Constructs and transmits VMT Room pose packets."""
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            pitch, yaw, roll = float(euler[0]), float(euler[1]), float(euler[2])

            bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

            # /VMT/Room/Unity: (int)idx, (int)enable, (float)timeoffset, (float)x, y, z, qx, qy, qz, qw
            if self.broadcast_quaternion:
                msg_quat = osc_message_builder.OscMessageBuilder(address="/VMT/Room/Unity")
                msg_quat.add_arg(int(index))
                msg_quat.add_arg(int(enable))
                msg_quat.add_arg(0.0)  # timeoffset
                msg_quat.add_arg(x)
                msg_quat.add_arg(y)
                msg_quat.add_arg(z)
                msg_quat.add_arg(qx)
                msg_quat.add_arg(qy)
                msg_quat.add_arg(qz)
                msg_quat.add_arg(qw)
                bundle.add_content(msg_quat.build())

            # /VMT/Room/UEuler: (int)idx, (int)enable, (float)timeoffset, (float)x, y, z, rx, ry, rz
            if self.broadcast_euler:
                msg_euler = osc_message_builder.OscMessageBuilder(address="/VMT/Room/UEuler")
                msg_euler.add_arg(int(index))
                msg_euler.add_arg(int(enable))
                msg_euler.add_arg(0.0)  # timeoffset
                msg_euler.add_arg(x)
                msg_euler.add_arg(y)
                msg_euler.add_arg(z)
                msg_euler.add_arg(pitch)  # rx
                msg_euler.add_arg(yaw)    # ry
                msg_euler.add_arg(roll)   # rz
                bundle.add_content(msg_euler.build())

            self.client.send(bundle.build())
            self.packets_sent += 1
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_input_trigger(self, index: int, trigger_idx: int, value: float) -> bool:
        """
        Sends analog trigger pull value (0.0 to 1.0).
        trigger_idx: 0 = Main Trigger (Pinch / Point), 1 = Grip (Fist)
        """
        try:
            msg = osc_message_builder.OscMessageBuilder(address="/VMT/Input/Trigger")
            msg.add_arg(int(index))
            msg.add_arg(int(trigger_idx))
            msg.add_arg(0.0)  # timeoffset
            msg.add_arg(float(max(0.0, min(1.0, value))))
            self.client.send(msg.build())
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_input_button(self, index: int, button_idx: int, value: int) -> bool:
        """
        Sends digital button state (1 = Pressed, 0 = Released).
        button_idx: 0 = System / Trigger Click, 1 = Application Menu / Button A/X, 2 = Grip Click, 3 = Button B/Y
        """
        try:
            msg = osc_message_builder.OscMessageBuilder(address="/VMT/Input/Button")
            msg.add_arg(int(index))
            msg.add_arg(int(button_idx))
            msg.add_arg(0.0)  # timeoffset
            msg.add_arg(int(1 if value else 0))
            self.client.send(msg.build())
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_input_joystick(self, index: int, joystick_idx: int, x: float, y: float) -> bool:
        """
        Sends joystick / thumbstick coordinates (-1.0 to 1.0).
        joystick_idx: 0 = Trackpad, 1 = Thumbstick
        """
        try:
            msg = osc_message_builder.OscMessageBuilder(address="/VMT/Input/Joystick")
            msg.add_arg(int(index))
            msg.add_arg(int(joystick_idx))
            msg.add_arg(0.0)  # timeoffset
            msg.add_arg(float(max(-1.0, min(1.0, x))))
            msg.add_arg(float(max(-1.0, min(1.0, y))))
            self.client.send(msg.build())
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_input_joystick_touch(self, index: int, joystick_idx: int, value: int) -> bool:
        """Sends thumbstick touch status (1 = touched, 0 = released)."""
        try:
            msg = osc_message_builder.OscMessageBuilder(address="/VMT/Input/Joystick/Touch")
            msg.add_arg(int(index))
            msg.add_arg(int(joystick_idx))
            msg.add_arg(0.0)
            msg.add_arg(int(1 if value else 0))
            self.client.send(msg.build())
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_hand_pose(
        self,
        handedness: str,
        position: Tuple[float, float, float],
        quaternion: Tuple[float, float, float, float],
        euler_deg: Tuple[float, float, float],
        gesture_name: str = "None",
        landmarks: Optional[List[Any]] = None,
        curls: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Broadcasts 60 Hz 6-DOF hand pose, analog triggers, grips, buttons, and
        pointing-tilt locomotion thumbstick to SteamVR VMT.

        :param handedness: "Left" or "Right"
        :param position: (x, y, z) in SteamVR playspace meters
        :param quaternion: (qw, qx, qy, qz) orientation
        :param euler_deg: (pitch, yaw, roll) in degrees
        :param gesture_name: GestureDrive classified pose name
        :param landmarks: 21 3D skeleton points (metric or normalized)
        :param curls: Dictionary of finger curl ratios (Thumb, Index, Middle, etc.)
        :param timestamp: Monotonic timestamp in seconds
        """
        if timestamp is None:
            timestamp = time.time()

        side = "Left" if str(handedness).lower() == "left" else "Right"
        state = self.hands[side]

        # Convert quaternion from (qw, qx, qy, qz) to VMT's (qx, qy, qz, qw)
        qw, qx, qy, qz = quaternion
        vmt_quat = (qx, qy, qz, qw)

        summary = {
            "side": side,
            "index": state.index,
            "role": state.role,
            "role_name": state.role.name,
            "pos": position,
            "euler": euler_deg,
            "trigger": 0.0,
            "trigger_click": 0,
            "grip": 0.0,
            "grip_button": 0,
            "button_a": 0,
            "button_b": 0,
            "joy_x": 0.0,
            "joy_y": 0.0,
            "joy_touch": 0,
            "sent": False,
        }
        self.last_summary[side] = summary

        if not self.enabled:
            return summary

        # 60 Hz rate limiter
        if (timestamp - state.last_send_time) < self.min_interval_sec:
            return summary

        # -------------------------------------------------------------
        # 1. Analog Index Trigger (0.0 -> 1.0 Continuous Pinch Distance)
        # -------------------------------------------------------------
        trigger_val = 0.0
        if landmarks and len(landmarks) >= 21:
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            wrist = landmarks[WRIST]
            middle_mcp = landmarks[MIDDLE_MCP]

            scale = max(0.01, dist_3d(wrist, middle_mcp))
            pinch_dist_norm = dist_3d(thumb_tip, index_tip) / scale

            # Continuous mapping:
            # pinch_dist_norm >= 0.65 -> 0.0 (wide open)
            # pinch_dist_norm <= 0.16 -> 1.0 (touching)
            if pinch_dist_norm >= 0.65:
                pinch_analog = 0.0
            elif pinch_dist_norm <= 0.16:
                pinch_analog = 1.0
            else:
                pinch_analog = (0.65 - pinch_dist_norm) / (0.65 - 0.16)
            trigger_val = pinch_analog

        if curls and "Index" in curls:
            trigger_val = max(trigger_val, curls["Index"])

        if gesture_name in ("Pinch", "OK"):
            trigger_val = max(trigger_val, 1.0)

        trigger_val = round(max(0.0, min(1.0, trigger_val)), 3)
        trigger_click = 1 if trigger_val >= 0.88 else 0

        # -------------------------------------------------------------
        # 2. Grip Button: Closed Fist & 3-Finger Squeeze
        # -------------------------------------------------------------
        grip_val = 0.0
        if gesture_name == "Fist":
            grip_val = 1.0
        elif curls:
            m_curl = curls.get("Middle", 0.0)
            r_curl = curls.get("Ring", 0.0)
            p_curl = curls.get("Pinky", 0.0)
            avg_grip_curl = (m_curl + r_curl + p_curl) / 3.0
            if avg_grip_curl >= 0.70:
                grip_val = 1.0
            elif avg_grip_curl <= 0.35:
                grip_val = 0.0
            else:
                grip_val = (avg_grip_curl - 0.35) / (0.70 - 0.35)

        grip_val = round(max(0.0, min(1.0, grip_val)), 3)
        grip_button = 1 if grip_val >= 0.80 else 0

        # -------------------------------------------------------------
        # 3. A/B or X/Y Buttons: Thumbs Up (B/Y) & Two-Finger Tap (A/X)
        # -------------------------------------------------------------
        # Button B (Right) / Button Y (Left) [button index 3]
        btn_b_val = 1 if gesture_name == "Thumbs Up" else 0

        # Button A (Right) / Button X (Left) [button index 1]
        # Triggered by Peace sign or two-finger tap (Index Tip & Middle Tip touching)
        two_finger_tap = False
        if landmarks and len(landmarks) >= 21:
            index_tip = landmarks[INDEX_TIP]
            middle_tip = landmarks[MIDDLE_TIP]
            scale = max(0.01, dist_3d(landmarks[WRIST], landmarks[MIDDLE_MCP]))
            tap_dist = dist_3d(index_tip, middle_tip) / scale
            if tap_dist <= 0.22:
                if curls and curls.get("Index", 1.0) < 0.40 and curls.get("Middle", 1.0) < 0.40:
                    two_finger_tap = True

        btn_a_val = 1 if (gesture_name == "Peace" or two_finger_tap) else 0

        # -------------------------------------------------------------
        # 4. Locomotion / Thumbstick: Natural Hand Tilt Angle
        # -------------------------------------------------------------
        # Eligible for tilt locomotion when hand is not a tight closed Fist
        is_locomotion_eligible = (gesture_name != "Fist")
        if curls:
            four_curl = (curls.get("Index", 0.0) + curls.get("Middle", 0.0) +
                         curls.get("Ring", 0.0) + curls.get("Pinky", 0.0)) * 0.25
            if four_curl > 0.80:
                is_locomotion_eligible = False

        pitch, yaw, roll = euler_deg
        deadzone = self.locomotion_deadzone_deg
        max_tilt = self.locomotion_max_tilt_deg

        if is_locomotion_eligible:
            # Pitch tilt: tilting down/forward (negative pitch) -> Move Forward (+Y)
            # Tilting up/backward (positive pitch) -> Move Backward (-Y)
            if abs(pitch) > deadzone:
                mag_p = min(1.0, (abs(pitch) - deadzone) / (max_tilt - deadzone))
                target_joy_y = -math.copysign(mag_p, pitch)
            else:
                target_joy_y = 0.0

            # Roll tilt: tilting hand right (positive roll) -> Strafe Right (+X)
            # Tilting hand left (negative roll) -> Strafe Left (-X)
            if abs(roll) > deadzone:
                mag_r = min(1.0, (abs(roll) - deadzone) / (max_tilt - deadzone))
                target_joy_x = math.copysign(mag_r, roll)
            else:
                target_joy_x = 0.0
        else:
            target_joy_x = 0.0
            target_joy_y = 0.0

        # Smooth low-pass thumbstick walking filter (fast stop when leveling hand)
        is_stopping = (target_joy_x == 0.0 and target_joy_y == 0.0)
        alpha = 0.75 if is_stopping else 0.50
        joy_x = round(alpha * target_joy_x + (1.0 - alpha) * state.last_joy_x, 3)
        joy_y = round(alpha * target_joy_y + (1.0 - alpha) * state.last_joy_y, 3)
        if abs(joy_x) < 0.015:
            joy_x = 0.0
        if abs(joy_y) < 0.015:
            joy_y = 0.0

        joy_touch = 1 if (abs(joy_x) > 0.0 or abs(joy_y) > 0.0) else 0

        # -------------------------------------------------------------
        # Dispatch 6-DOF Room Pose + Buttons/Triggers/Thumbstick
        # -------------------------------------------------------------
        enable_val = int(state.role)
        if self._send_raw_room_pose(state.index, enable_val, position, vmt_quat, euler_deg):
            state.last_send_time = timestamp
            state.last_pos = position
            state.last_euler = euler_deg
            summary["sent"] = True

        # Send Analog Trigger (trigger_idx=0)
        if abs(trigger_val - state.last_trigger) >= 0.015:
            self.send_input_trigger(state.index, trigger_idx=0, value=trigger_val)
            state.last_trigger = trigger_val

        # Send Digital Trigger Click (button_idx=0)
        if trigger_click != state.last_trigger_click:
            self.send_input_button(state.index, button_idx=0, value=trigger_click)
            state.last_trigger_click = trigger_click

        # Send Analog Grip (trigger_idx=1)
        if abs(grip_val - state.last_grip) >= 0.015:
            self.send_input_trigger(state.index, trigger_idx=1, value=grip_val)
            state.last_grip = grip_val

        # Send Digital Grip Click (button_idx=2)
        if grip_button != state.last_grip_button:
            self.send_input_button(state.index, button_idx=2, value=grip_button)
            state.last_grip_button = grip_button

        # Send Button A/X (button_idx=1)
        if btn_a_val != state.last_button_a:
            self.send_input_button(state.index, button_idx=1, value=btn_a_val)
            state.last_button_a = btn_a_val

        # Send Button B/Y (button_idx=3)
        if btn_b_val != state.last_button_b:
            self.send_input_button(state.index, button_idx=3, value=btn_b_val)
            state.last_button_b = btn_b_val

        # Send Locomotion Joystick (Trackpad=0, Thumbstick=1)
        if abs(joy_x - state.last_joy_x) >= 0.015 or abs(joy_y - state.last_joy_y) >= 0.015:
            self.send_input_joystick(state.index, joystick_idx=0, x=joy_x, y=joy_y)
            self.send_input_joystick(state.index, joystick_idx=1, x=joy_x, y=joy_y)
            state.last_joy_x = joy_x
            state.last_joy_y = joy_y

        # Send Joystick Touch
        if joy_touch != state.last_joy_touch:
            self.send_input_joystick_touch(state.index, joystick_idx=0, value=joy_touch)
            self.send_input_joystick_touch(state.index, joystick_idx=1, value=joy_touch)
            state.last_joy_touch = joy_touch

        summary["trigger"] = trigger_val
        summary["trigger_click"] = trigger_click
        summary["grip"] = grip_val
        summary["grip_button"] = grip_button
        summary["button_a"] = btn_a_val
        summary["button_b"] = btn_b_val
        summary["joy_x"] = joy_x
        summary["joy_y"] = joy_y
        summary["joy_touch"] = joy_touch

        return summary

    def reset_neutral(self):
        """Resets all trigger, grip, button, and thumbstick inputs and despawns virtual devices."""
        for side, state in self.hands.items():
            self.send_input_trigger(state.index, trigger_idx=0, value=0.0)
            self.send_input_trigger(state.index, trigger_idx=1, value=0.0)
            self.send_input_button(state.index, button_idx=0, value=0)
            self.send_input_button(state.index, button_idx=1, value=0)
            self.send_input_button(state.index, button_idx=2, value=0)
            self.send_input_button(state.index, button_idx=3, value=0)
            self.send_input_joystick(state.index, joystick_idx=0, x=0.0, y=0.0)
            self.send_input_joystick(state.index, joystick_idx=1, x=0.0, y=0.0)
            self.send_input_joystick_touch(state.index, joystick_idx=0, value=0)
            self.send_input_joystick_touch(state.index, joystick_idx=1, value=0)

            self._send_raw_room_pose(
                index=state.index,
                enable=int(VMTDeviceRole.DISABLED),
                pos=(0.0, 0.0, 0.0),
                quat=(0.0, 0.0, 0.0, 1.0),
                euler=(0.0, 0.0, 0.0),
            )
            state.reset()
        self.last_summary.clear()
