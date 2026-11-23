"""
vr/osc_vrchat.py - Day 3 VR Upgrade: VRChat OSC Integration (Zero-Driver Hand Tracking).
Streams live finger flexion (curl & spread) and hand gesture codes over UDP/OSC
to VRChat on 127.0.0.1:9000 using python-osc without requiring custom SteamVR drivers.
"""

import time
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any, Union

from pythonosc import udp_client
from pythonosc import osc_bundle_builder
from pythonosc import osc_message_builder

from vision.landmarks import (
    WRIST,
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
    dist_3d,
    calculate_angle_3d_deg,
    angle_between_vectors_3d_deg,
)
from vision.filters import LowPassFilter


FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Pinky")


class VRChatGesture(IntEnum):
    """Standard VRChat Avatar 3.0 GestureLeft / GestureRight integer parameters (0..7)."""
    NEUTRAL = 0
    FIST = 1
    HAND_OPEN = 2
    FINGER_POINT = 3
    VICTORY = 4
    ROCK_N_ROLL = 5
    HAND_GUN = 6
    THUMBS_UP = 7


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _remap01(val: float, low: float, high: float) -> float:
    """Linearly maps val from [low, high] to [0.0, 1.0]. Supports inverted ranges (low > high)."""
    if abs(high - low) < 1e-6:
        return 0.0
    return _clamp01((val - low) / (high - low))


def _vec3_sub(a, b) -> Tuple[float, float, float]:
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def compute_finger_curls_and_spreads(skeleton_3d: List[Any]) -> Dict[str, Dict[str, float]]:
    """
    Computes continuous normalized flexion curl [0.0=open, 1.0=curled] and
    splay spread [0.0=together, 1.0=wide] for all 5 fingers from a 21-point 3D hand skeleton.
    Metric 3D points (meters) give best aspect-ratio-invariant joint angles.
    """
    zeros = {name: 0.0 for name in FINGER_NAMES}
    if not skeleton_3d or len(skeleton_3d) < 21:
        return {"curls": dict(zeros), "spreads": dict(zeros)}

    wrist = skeleton_3d[WRIST]
    hand_scale = max(0.01, dist_3d(wrist, skeleton_3d[MIDDLE_MCP]))

    # 1. Non-thumb curls from combined MCP + PIP + 0.5*DIP 3D joint flexion
    finger_chains = {
        "Index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        "Middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        "Ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        "Pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    }

    curls: Dict[str, float] = {}
    for name, (mcp_i, pip_i, dip_i, tip_i) in finger_chains.items():
        mcp = skeleton_3d[mcp_i]
        pip = skeleton_3d[pip_i]
        dip = skeleton_3d[dip_i]
        tip = skeleton_3d[tip_i]

        ang_mcp = calculate_angle_3d_deg(wrist, mcp, pip)
        ang_pip = calculate_angle_3d_deg(mcp, pip, dip)
        ang_dip = calculate_angle_3d_deg(pip, dip, tip)

        total_bend_deg = (180.0 - ang_mcp) + (180.0 - ang_pip) + 0.5 * (180.0 - ang_dip)
        curls[name] = _remap01(total_bend_deg, 22.0, 195.0)

    # 2. Thumb curl combines IP joint bend with palm opposition distance ratio
    thumb_ip_ang = calculate_angle_3d_deg(
        skeleton_3d[THUMB_MCP], skeleton_3d[THUMB_IP], skeleton_3d[THUMB_TIP]
    )
    thumb_ip_curl = _remap01(thumb_ip_ang, 170.0, 105.0)

    thumb_opp_ratio = dist_3d(skeleton_3d[THUMB_TIP], skeleton_3d[RING_MCP]) / hand_scale
    thumb_opp_curl = _remap01(thumb_opp_ratio, 1.35, 0.50)

    curls["Thumb"] = _clamp01(0.45 * thumb_ip_curl + 0.55 * thumb_opp_curl)

    # 3. Adjacent finger splay spreads using proximal ray angles (MCP -> PIP)
    ray_thumb = _vec3_sub(skeleton_3d[THUMB_IP], skeleton_3d[THUMB_CMC])
    ray_index = _vec3_sub(skeleton_3d[INDEX_PIP], skeleton_3d[INDEX_MCP])
    ray_middle = _vec3_sub(skeleton_3d[MIDDLE_PIP], skeleton_3d[MIDDLE_MCP])
    ray_ring = _vec3_sub(skeleton_3d[RING_PIP], skeleton_3d[RING_MCP])
    ray_pinky = _vec3_sub(skeleton_3d[PINKY_PIP], skeleton_3d[PINKY_MCP])

    s_ti = _remap01(angle_between_vectors_3d_deg(ray_thumb, ray_index), 20.0, 58.0)
    s_im = _remap01(angle_between_vectors_3d_deg(ray_index, ray_middle), 6.0, 28.0)
    s_mr = _remap01(angle_between_vectors_3d_deg(ray_middle, ray_ring), 5.0, 26.0)
    s_rp = _remap01(angle_between_vectors_3d_deg(ray_ring, ray_pinky), 7.0, 32.0)

    raw_spreads = {
        "Thumb": s_ti,
        "Index": s_im,
        "Middle": 0.5 * (s_im + s_mr),
        "Ring": 0.5 * (s_mr + s_rp),
        "Pinky": s_rp,
    }

    # Attenuate spread noise when a finger is tightly curled into palm
    spreads = {
        name: _clamp01(raw_spreads[name] * (1.0 - 0.75 * curls[name]))
        for name in FINGER_NAMES
    }

    return {"curls": curls, "spreads": spreads}


def map_hand_to_vrchat_gesture(
    gesture_name: str,
    curls: Dict[str, float]
) -> Tuple[VRChatGesture, float]:
    """
    Maps GestureDrive pose + finger curls to VRChat Avatar 3.0 GestureLeft/GestureRight (0..7)
    and continuous GestureWeight (0.0..1.0).
    """
    four_avg = (curls["Index"] + curls["Middle"] + curls["Ring"] + curls["Pinky"]) * 0.25
    thumb_curl = curls["Thumb"]

    mapping = {
        "Fist": VRChatGesture.FIST,
        "Open Palm": VRChatGesture.HAND_OPEN,
        "Peace": VRChatGesture.VICTORY,
        "Rock On": VRChatGesture.ROCK_N_ROLL,
        "Thumbs Up": VRChatGesture.THUMBS_UP,
        "Pinch": VRChatGesture.FIST,
        "OK": VRChatGesture.HAND_OPEN,
    }

    if gesture_name == "Pointing":
        # Thumb raised while pointing index -> HandGun (6), otherwise FingerPoint (3)
        vrc_gesture = VRChatGesture.HAND_GUN if thumb_curl < 0.42 else VRChatGesture.FINGER_POINT
    elif gesture_name in mapping:
        vrc_gesture = mapping[gesture_name]
    else:
        # Fallback inference purely from curl flexion state
        if four_avg > 0.70 and thumb_curl > 0.55:
            vrc_gesture = VRChatGesture.FIST
        elif four_avg < 0.25:
            vrc_gesture = VRChatGesture.HAND_OPEN
        else:
            vrc_gesture = VRChatGesture.NEUTRAL

    # GestureWeight reflects trigger/grip analog squeeze intensity
    if vrc_gesture in (VRChatGesture.FIST, VRChatGesture.FINGER_POINT, VRChatGesture.HAND_GUN):
        weight = max(curls["Index"], four_avg) if gesture_name == "Pinch" else four_avg
    elif vrc_gesture == VRChatGesture.HAND_OPEN:
        weight = _clamp01(1.0 - four_avg)
    else:
        weight = _clamp01(0.5 * curls["Index"] + 0.5 * four_avg)

    return vrc_gesture, _clamp01(weight)


class _HandOSCState:
    """Maintains per-hand low-pass smoothed curls, spreads, and cached OSC values."""
    def __init__(self, smooth_alpha: float = 0.45):
        self.alpha = smooth_alpha
        self.curl_filters = {name: LowPassFilter(smooth_alpha) for name in FINGER_NAMES}
        self.spread_filters = {name: LowPassFilter(smooth_alpha) for name in FINGER_NAMES}
        self.weight_filter = LowPassFilter(smooth_alpha)
        self.last_sent: Dict[str, Union[int, float]] = {}
        self.last_heartbeat_time: float = 0.0

    def reset(self):
        for f in self.curl_filters.values():
            f.reset()
        for f in self.spread_filters.values():
            f.reset()
        self.weight_filter.reset()
        self.last_sent.clear()
        self.last_heartbeat_time = 0.0


class VRChatOSCClient:
    """
    Zero-driver VRChat OSC transmitter using python-osc on 127.0.0.1:9000.
    Broadcasts /avatar/parameters/GestureLeft & GestureRight plus skeletal curl & spread floats.
    """
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        max_send_hz: float = 45.0,
        change_epsilon: float = 0.015,
        heartbeat_sec: float = 1.5,
    ):
        self.host = host
        self.port = int(port)
        self.min_interval_sec = 1.0 / max(1.0, max_send_hz)
        self.change_epsilon = change_epsilon
        self.heartbeat_sec = heartbeat_sec

        self.enabled = True
        self.client = udp_client.SimpleUDPClient(self.host, self.port)

        self._hand_states: Dict[str, _HandOSCState] = {
            "Left": _HandOSCState(),
            "Right": _HandOSCState(),
        }
        self._last_send_time_by_hand: Dict[str, float] = {"Left": 0.0, "Right": 0.0}
        self.packets_sent: int = 0
        self.last_error: Optional[str] = None
        self.last_summary: Dict[str, Dict[str, Any]] = {}

    def toggle_enabled(self) -> bool:
        """Toggles VRChat OSC streaming on or off."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.reset_neutral()
        print(f"[VRChat OSC] Streaming {'ENABLED' if self.enabled else 'PAUSED'} -> {self.host}:{self.port}")
        return self.enabled

    def _send_bundle(self, params: Dict[str, Union[int, float]]) -> bool:
        """Packs OSC parameter updates into one atomic UDP OscBundle."""
        if not params:
            return False
        try:
            builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            for addr, val in params.items():
                msg = osc_message_builder.OscMessageBuilder(address=addr)
                msg.add_arg(val)
                builder.add_content(msg.build())
            self.client.send(builder.build())
            self.packets_sent += 1
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def send_hand_state(
        self,
        handedness: str,
        skeleton_3d: List[Any],
        gesture_name: str = "None",
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes finger curls/spreads and VRChat gesture code for one hand,
        smooths values, and streams changed parameters via OSC to VRChat.
        """
        if timestamp is None:
            timestamp = time.time()

        side = "Left" if str(handedness).lower() == "left" else "Right"
        state = self._hand_states[side]

        raw_kin = compute_finger_curls_and_spreads(skeleton_3d)
        smooth_curls = {
            name: state.curl_filters[name].filter(raw_kin["curls"][name])
            for name in FINGER_NAMES
        }
        smooth_spreads = {
            name: state.spread_filters[name].filter(raw_kin["spreads"][name])
            for name in FINGER_NAMES
        }

        vrc_gesture, raw_weight = map_hand_to_vrchat_gesture(gesture_name, smooth_curls)
        smooth_weight = state.weight_filter.filter(raw_weight)

        # Build standard VRChat OSC parameter dictionary for this hand
        desired: Dict[str, Union[int, float]] = {
            f"/avatar/parameters/Gesture{side}": int(vrc_gesture.value),
            f"/avatar/parameters/Gesture{side}Weight": round(smooth_weight, 3),
        }
        for fname in FINGER_NAMES:
            desired[f"/avatar/parameters/{side}FingerCurl{fname}"] = round(smooth_curls[fname], 3)
            desired[f"/avatar/parameters/{side}FingerSpread{fname}"] = round(smooth_spreads[fname], 3)

        # Stream standard VRChat /input/ interaction commands (natively supported by VRChat)
        is_grab = 1 if (gesture_name in ("Fist", "Pinch") or smooth_weight > 0.72) else 0
        is_use = 1 if (gesture_name in ("Pinch", "Pointing", "Hand Gun")) else 0
        desired[f"/input/Grab{side}"] = is_grab
        desired[f"/input/Use{side}"] = is_use
        if side == "Right":
            desired["/input/GrabAxisRight"] = round(smooth_weight, 3)
            desired["/input/UseAxisRight"] = round(smooth_curls["Index"], 3)

        summary = {
            "side": side,
            "gesture": vrc_gesture,
            "gesture_name": vrc_gesture.name,
            "gesture_code": int(vrc_gesture.value),
            "weight": smooth_weight,
            "curls": smooth_curls,
            "spreads": smooth_spreads,
            "osc_sent": False,
        }
        self.last_summary[side] = summary

        if not self.enabled:
            return summary

        if (timestamp - self._last_send_time_by_hand[side]) < self.min_interval_sec:
            return summary

        force_heartbeat = (timestamp - state.last_heartbeat_time) >= self.heartbeat_sec
        payload: Dict[str, Union[int, float]] = {}

        for addr, val in desired.items():
            prev = state.last_sent.get(addr)
            if force_heartbeat or prev is None:
                payload[addr] = val
            elif isinstance(val, int) and val != prev:
                payload[addr] = val
            elif isinstance(val, float) and abs(val - prev) >= self.change_epsilon:
                payload[addr] = val

        if payload:
            if self._send_bundle(payload):
                self._last_send_time_by_hand[side] = timestamp
                state.last_sent.update(payload)
                if force_heartbeat:
                    state.last_heartbeat_time = timestamp
                summary["osc_sent"] = True

        return summary

    def send_locomotion(self, x: float, y: float) -> bool:
        """
        Streams VRChat locomotion input axes (/input/Vertical, /input/Horizontal).
        x: Strafe (-1.0 left to +1.0 right)
        y: Move (-1.0 backward to +1.0 forward)
        """
        if not self.enabled:
            return False
        try:
            bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            msg_vert = osc_message_builder.OscMessageBuilder(address="/input/Vertical")
            msg_vert.add_arg(float(max(-1.0, min(1.0, y))))
            bundle.add_content(msg_vert.build())

            msg_horiz = osc_message_builder.OscMessageBuilder(address="/input/Horizontal")
            msg_horiz.add_arg(float(max(-1.0, min(1.0, x))))
            bundle.add_content(msg_horiz.build())

            self.client.send(bundle.build())
            return True
        except Exception:
            return False

    def send_look(self, look_x: float) -> bool:
        """
        Streams VRChat rotation input axis (/input/LookHorizontal).
        look_x: Rotation / Look (-1.0 rotate left to +1.0 rotate right)
        """
        if not self.enabled:
            return False
        try:
            msg = osc_message_builder.OscMessageBuilder(address="/input/LookHorizontal")
            msg.add_arg(float(max(-1.0, min(1.0, look_x))))
            self.client.send(msg.build())
            return True
        except Exception:
            return False

    def reset_neutral(self):
        """Resets both Left and Right VRChat hands back to Neutral (0) and uncurled state."""
        self.send_locomotion(0.0, 0.0)
        self.send_look(0.0)
        neutral_params: Dict[str, Union[int, float]] = {}
        for side in ("Left", "Right"):
            neutral_params[f"/avatar/parameters/Gesture{side}"] = int(VRChatGesture.NEUTRAL.value)
            neutral_params[f"/avatar/parameters/Gesture{side}Weight"] = 0.0
            neutral_params[f"/input/Grab{side}"] = 0
            neutral_params[f"/input/Use{side}"] = 0
            for fname in FINGER_NAMES:
                neutral_params[f"/avatar/parameters/{side}FingerCurl{fname}"] = 0.0
                neutral_params[f"/avatar/parameters/{side}FingerSpread{fname}"] = 0.0
            self._hand_states[side].reset()

        neutral_params["/input/GrabAxisRight"] = 0.0
        neutral_params["/input/UseAxisRight"] = 0.0

        self._send_bundle(neutral_params)
        self.last_summary.clear()

