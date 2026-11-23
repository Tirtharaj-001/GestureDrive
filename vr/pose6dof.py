"""
vr/pose6dof.py - Day 2 VR Upgrade: 6-DOF Hand Orientation, 3D Spatial Smoothing & Kinematics.
Computes 3D rotation quaternions (qw, qx, qy, qz) from the palm plane (Wrist, Index MCP, Pinky MCP),
applies 3D One-Euro spatial filtering + Slerp quaternion smoothing independently per hand,
and estimates linear (m/s) & angular (rad/s) velocities for VR physics.
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any

from vision.landmarks import WRIST, INDEX_MCP, PINKY_MCP
from vision.filters import (
    LowPassFilter,
    OneEuroFilter3D,
    OneEuroQuaternionFilter,
    Quaternion,
    _one_euro_alpha,
    quat_normalize,
)

Vec3 = Tuple[float, float, float]


def _vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_normalize(v: Vec3) -> Optional[Vec3]:
    n = _vec_norm(v)
    if n < 1e-7:
        return None
    inv = 1.0 / n
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def quat_conjugate(q: Quaternion) -> Quaternion:
    """Returns inverse of a unit quaternion (w, -x, -y, -z)."""
    return (q[0], -q[1], -q[2], -q[3])


def quat_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Hamilton product q1 * q2 in (w, x, y, z) convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def rotation_matrix_to_quaternion(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Quaternion:
    """
    Converts a right-handed orthonormal rotation matrix R = [x_axis | y_axis | z_axis]
    to a unit quaternion (qw, qx, qy, qz) using Shepperd's numerically stable method.
    """
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(max(1e-8, 1.0 + m00 - m11 - m22))
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(max(1e-8, 1.0 + m11 - m00 - m22))
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(max(1e-8, 1.0 + m22 - m00 - m11))
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return quat_normalize((qw, qx, qy, qz))


def quaternion_to_rotation_matrix(q: Quaternion) -> Tuple[Vec3, Vec3, Vec3]:
    """Converts a unit quaternion (w, x, y, z) into basis columns (x_axis, y_axis, z_axis)."""
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    x_axis = (1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy))
    y_axis = (2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx))
    z_axis = (2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy))
    return x_axis, y_axis, z_axis


def quaternion_to_euler_deg(q: Quaternion) -> Tuple[float, float, float]:
    """
    Converts unit quaternion to intuitive VR (pitch_deg, yaw_deg, roll_deg):
    - Pitch (+): fingers tilted up toward ceiling
    - Yaw (+):   hand turned right
    - Roll (+):  hand twisted clockwise (right side down)
    """
    x_axis, y_axis, z_axis = quaternion_to_rotation_matrix(q)

    # Forward vector z_axis = (zx, zy, zz)
    sin_pitch = max(-1.0, min(1.0, z_axis[1]))
    pitch_rad = math.asin(sin_pitch)

    # Guard against gimbal lock when pointing straight up or straight down (+/- 90 deg)
    if abs(sin_pitch) < 0.999:
        yaw_rad = math.atan2(z_axis[0], z_axis[2])
        roll_rad = math.atan2(-x_axis[1], y_axis[1])
    else:
        yaw_rad = math.atan2(-x_axis[2], x_axis[0])
        roll_rad = 0.0

    return (
        math.degrees(pitch_rad),
        math.degrees(yaw_rad),
        math.degrees(roll_rad),
    )


def compute_palm_orientation_quaternion(
    vr_skeleton: List[Any],
    handedness: str = "Right"
) -> Tuple[Quaternion, Tuple[Vec3, Vec3, Vec3]]:
    """
    Computes a 3D rotation quaternion (qw, qx, qy, qz) and local basis axes from the palm plane
    defined by (WRIST, INDEX_MCP, PINKY_MCP) in VR space (+X Right, +Y Up, +Z Forward).

    Neutral identity pose (0, 0, 0 deg):
    - Fingers extending forward (+Z)
    - Back of palm facing up (+Y)
    - Palm span across left-to-right (+X)
    """
    default_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if not vr_skeleton or len(vr_skeleton) < 21:
        return (1.0, 0.0, 0.0, 0.0), default_axes

    w_pt = vr_skeleton[WRIST]
    i_pt = vr_skeleton[INDEX_MCP]
    p_pt = vr_skeleton[PINKY_MCP]

    w_vec = (w_pt.x, w_pt.y, w_pt.z)
    i_vec = (i_pt.x, i_pt.y, i_pt.z)
    p_vec = (p_pt.x, p_pt.y, p_pt.z)

    # Forward axis points from Wrist through the midpoint of Index & Pinky knuckles
    mid_knuckle = (
        0.5 * (i_vec[0] + p_vec[0]),
        0.5 * (i_vec[1] + p_vec[1]),
        0.5 * (i_vec[2] + p_vec[2]),
    )
    z_axis = _vec_normalize(_vec_sub(mid_knuckle, w_vec))
    if z_axis is None:
        return (1.0, 0.0, 0.0, 0.0), default_axes

    # Across-palm vector aligned toward +X (Right) for both hands
    if handedness.lower() == "left":
        across_raw = _vec_sub(i_vec, p_vec)  # Left hand: Pinky -> Index points Right
    else:
        across_raw = _vec_sub(p_vec, i_vec)  # Right hand: Index -> Pinky points Right

    # Build dorsal (back-of-hand Up) axis perpendicular to palm plane via Z x X = Y
    y_axis = _vec_normalize(_vec_cross(z_axis, across_raw))
    if y_axis is None:
        return (1.0, 0.0, 0.0, 0.0), default_axes

    # Re-orthogonalize Right axis strictly via Y x Z = X so det(R) = +1 is guaranteed
    x_axis = _vec_normalize(_vec_cross(y_axis, z_axis))
    if x_axis is None:
        return (1.0, 0.0, 0.0, 0.0), default_axes

    q_raw = rotation_matrix_to_quaternion(x_axis, y_axis, z_axis)
    return q_raw, (x_axis, y_axis, z_axis)


def compute_angular_velocity(q_prev: Quaternion, q_curr: Quaternion, dt: float) -> Tuple[Vec3, float]:
    """
    Computes angular velocity vector (wx, wy, wz) in rad/s and scalar angular speed (rad/s)
    from two consecutive unit quaternions over timestep dt.
    """
    if dt <= 1e-5:
        return (0.0, 0.0, 0.0), 0.0

    # Relative world-frame rotation delta: dq = q_curr * q_prev^-1
    dq = quat_multiply(quat_normalize(q_curr), quat_conjugate(quat_normalize(q_prev)))
    w, x, y, z = dq

    # Force shortest rotation arc (w >= 0 -> angle in [0, pi])
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z

    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-6:
        return (0.0, 0.0, 0.0), 0.0

    angle_rad = 2.0 * math.atan2(sin_half, max(-1.0, min(1.0, w)))
    speed_rad = angle_rad / dt
    inv_axis = speed_rad / sin_half

    return (x * inv_axis, y * inv_axis, z * inv_axis), speed_rad


class _SingleHand6DOFState:
    """Maintains independent 3D position, Slerp quaternion, and velocity filters for one hand."""
    def __init__(self, freq: float = 60.0):
        self.pos_filter = OneEuroFilter3D(freq=freq, mincutoff=1.0, beta=15.0, dcutoff=1.0)
        self.rot_filter = OneEuroQuaternionFilter(freq=freq, mincutoff=1.0, beta=2.0, dcutoff=1.0)

        self._lin_vel_filters = (LowPassFilter(), LowPassFilter(), LowPassFilter())
        self._ang_vel_filters = (LowPassFilter(), LowPassFilter(), LowPassFilter())

        self.prev_time: Optional[float] = None
        self.prev_pos: Optional[Vec3] = None
        self.prev_quat: Optional[Quaternion] = None
        self.last_seen_time: float = 0.0

    def reset(self):
        self.pos_filter.reset()
        self.rot_filter.reset()
        for f in self._lin_vel_filters:
            f.reset()
        for f in self._ang_vel_filters:
            f.reset()
        self.prev_time = None
        self.prev_pos = None
        self.prev_quat = None

    def update(self, raw_pos: Vec3, raw_quat: Quaternion, timestamp: float) -> Dict[str, Any]:
        self.last_seen_time = timestamp

        smooth_pos = self.pos_filter.filter(raw_pos[0], raw_pos[1], raw_pos[2], timestamp=timestamp)
        smooth_quat = self.rot_filter.filter(raw_quat, timestamp=timestamp)

        # Estimate linear & angular velocities from smoothed pose deltas
        if self.prev_time is None or self.prev_pos is None or self.prev_quat is None:
            lin_vel = (0.0, 0.0, 0.0)
            ang_vel = (0.0, 0.0, 0.0)
        else:
            dt = timestamp - self.prev_time
            if dt > 0.4:
                # Stale gap reset to prevent phantom velocity spike
                for f in self._lin_vel_filters:
                    f.reset()
                for f in self._ang_vel_filters:
                    f.reset()
                lin_vel = (0.0, 0.0, 0.0)
                ang_vel = (0.0, 0.0, 0.0)
            else:
                dt_safe = max(1.0 / 120.0, dt)
                raw_vx = (smooth_pos[0] - self.prev_pos[0]) / dt_safe
                raw_vy = (smooth_pos[1] - self.prev_pos[1]) / dt_safe
                raw_vz = (smooth_pos[2] - self.prev_pos[2]) / dt_safe

                raw_ang_vel, _ = compute_angular_velocity(self.prev_quat, smooth_quat, dt_safe)

                # High-cutoff low-pass preserves fast throws/swings while rejecting frame jitter
                a_lin = _one_euro_alpha(14.0, dt_safe)
                a_ang = _one_euro_alpha(12.0, dt_safe)
                lin_vel = (
                    self._lin_vel_filters[0].filter(raw_vx, a_lin),
                    self._lin_vel_filters[1].filter(raw_vy, a_lin),
                    self._lin_vel_filters[2].filter(raw_vz, a_lin),
                )
                ang_vel = (
                    self._ang_vel_filters[0].filter(raw_ang_vel[0], a_ang),
                    self._ang_vel_filters[1].filter(raw_ang_vel[1], a_ang),
                    self._ang_vel_filters[2].filter(raw_ang_vel[2], a_ang),
                )

        self.prev_time = timestamp
        self.prev_pos = smooth_pos
        self.prev_quat = smooth_quat

        lin_speed = _vec_norm(lin_vel)
        ang_speed_rad = _vec_norm(ang_vel)
        ang_speed_deg = math.degrees(ang_speed_rad)

        basis_axes = quaternion_to_rotation_matrix(smooth_quat)
        euler_deg = quaternion_to_euler_deg(smooth_quat)

        return {
            "position_m": smooth_pos,
            "quaternion": smooth_quat,
            "raw_quaternion": raw_quat,
            "euler_deg": euler_deg,
            "basis_axes": basis_axes,
            "linear_velocity": lin_vel,
            "linear_speed": lin_speed,
            "angular_velocity": ang_vel,
            "angular_speed_rad": ang_speed_rad,
            "angular_speed_deg": ang_speed_deg,
        }


class Hand6DOFTracker:
    """
    Dual-hand 6-DOF pose tracker maintaining separate 3D filters and velocity states
    for both Left and Right hands simultaneously.
    """
    def __init__(self, freq: float = 60.0):
        self.freq = freq
        self._hand_states: Dict[str, _SingleHand6DOFState] = {}

    def update_hand(
        self,
        vr_palm_pos: Vec3,
        vr_skeleton: List[Any],
        handedness: str = "Right",
        timestamp: Optional[float] = None,
        track_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Updates smoothed 6-DOF position, quaternion, Euler angles, and velocities for one hand."""
        if timestamp is None:
            timestamp = time.time()

        hand_side = handedness if handedness in ("Left", "Right") else "Right"
        state_key = track_key or hand_side
        if state_key not in self._hand_states:
            self._hand_states[state_key] = _SingleHand6DOFState(freq=self.freq)

        raw_quat, _ = compute_palm_orientation_quaternion(vr_skeleton, handedness=hand_side)
        pose = self._hand_states[state_key].update(vr_palm_pos, raw_quat, timestamp)
        pose["handedness"] = hand_side
        pose["track_key"] = state_key
        return pose

    def prune_stale_hands(self, active_keys: set, timestamp: Optional[float] = None, max_age_sec: float = 0.45):
        """Resets filter state for hands absent longer than max_age_sec so re-entry starts fresh."""
        if timestamp is None:
            timestamp = time.time()
        for key, state in self._hand_states.items():
            if key not in active_keys and state.prev_time is not None:
                if (timestamp - state.last_seen_time) > max_age_sec:
                    state.reset()

    def reset(self):
        """Resets 6-DOF smoothing and velocity history for all tracked hands."""
        for state in self._hand_states.values():
            state.reset()
        self._hand_states.clear()
