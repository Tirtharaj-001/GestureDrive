"""
vision/filters.py - High-precision smoothing filters for hand landmark and cursor coordinates.
Implements the 1€ (One-Euro) Filter tuned for rock-solid stability and zero lag,
along with an Adaptive Pointer Precision curve for stable mid-air hand control.
"""

import math
import time
from typing import Tuple, Optional


Quaternion = Tuple[float, float, float, float]


def _one_euro_alpha(cutoff: float, dt: float) -> float:
    """Computes smoothing factor alpha from cutoff frequency (Hz) and timestep (s)."""
    tau = 1.0 / (2.0 * math.pi * max(1e-6, cutoff))
    return 1.0 / (1.0 + tau / dt)


def quat_normalize(q: Quaternion) -> Quaternion:
    """Normalizes a (w, x, y, z) quaternion to unit length."""
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return (1.0, 0.0, 0.0, 0.0)
    inv = 1.0 / norm
    return (w * inv, x * inv, y * inv, z * inv)


def quat_dot(q1: Quaternion, q2: Quaternion) -> float:
    """Computes dot product between two (w, x, y, z) quaternions."""
    return q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]


def quat_slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """
    Spherical Linear Interpolation (Slerp) between two unit quaternions along the shortest arc.
    Ensures antipodal continuity (flips q2 if dot < 0) and falls back to nlerp near 0 deg.
    """
    t = max(0.0, min(1.0, float(t)))
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)

    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2

    # Ensure shortest rotation path (q and -q represent the same orientation)
    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot

    # Fallback to normalized linear interpolation when orientations are nearly identical
    if dot > 0.9995:
        return quat_normalize((
            w1 + t * (w2 - w1),
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
            z1 + t * (z2 - z1),
        ))

    dot = max(-1.0, min(1.0, dot))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t

    s2 = math.sin(theta) / sin_theta_0
    s1 = math.cos(theta) - dot * s2

    return quat_normalize((
        s1 * w1 + s2 * w2,
        s1 * x1 + s2 * x2,
        s1 * y1 + s2 * y2,
        s1 * z1 + s2 * z2,
    ))


class LowPassFilter:
    """Standard first-order low-pass filter."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def reset(self):
        self.s = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter1D:
    """
    One-Euro Filter (1D) based on Casiez et al., 2012.
    Dynamically adjusts low-pass cutoff frequency based on movement speed:
    - At rest/slow speed (mincutoff=0.5Hz): heavily damps human hand tremor for rock-solid stillness.
    - At high speed (beta=0.03): dynamically increases cutoff for instant 1:1 responsive tracking.
    """
    def __init__(self, freq: float = 60.0, mincutoff: float = 0.5, beta: float = 0.030, dcutoff: float = 1.0):
        self.freq = max(1.0, freq)
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None
        self.last_x: Optional[float] = None

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.last_x = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        return _one_euro_alpha(cutoff, dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is None:
            timestamp = time.time()

        if self.last_time is None:
            self.last_time = timestamp
            self.last_x = x
            return self.x_filter.filter(x, alpha=1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 1e-5:
            dt = 1.0 / self.freq
        elif dt > 0.4:
            self.reset()
            self.last_time = timestamp
            self.last_x = x
            return self.x_filter.filter(x, alpha=1.0)

        # Calculate speed magnitude (derivative)
        dx = (x - self.last_x) / dt if self.last_x is not None else 0.0
        self.last_x = x

        # Smooth derivative
        a_d = self._alpha(self.dcutoff, dt)
        edx = self.dx_filter.filter(dx, a_d)

        # Dynamic cutoff frequency based on velocity
        cutoff = self.mincutoff + self.beta * abs(edx)

        # Smooth coordinate with dynamic cutoff
        a = self._alpha(cutoff, dt)
        return self.x_filter.filter(x, a)


class OneEuroFilter2D:
    """
    Pair of One-Euro filters for smooth 2D (X, Y) coordinate tracking.
    Tuned for rock-solid stability and zero lag.
    """
    def __init__(self, freq: float = 60.0, mincutoff: float = 0.5, beta: float = 0.030, dcutoff: float = 1.0):
        self.filter_x = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self.filter_y = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        if timestamp is None:
            timestamp = time.time()
        smoothed_x = self.filter_x.filter(x, timestamp)
        smoothed_y = self.filter_y.filter(y, timestamp)
        return smoothed_x, smoothed_y


class OneEuroFilter3D:
    """
    Triplet of One-Euro filters for smooth 3D (X, Y, Z) metric VR position tracking.
    Default tuning is scaled for meters (m/s speed range) rather than screen pixels.
    """
    def __init__(self, freq: float = 60.0, mincutoff: float = 1.0, beta: float = 15.0, dcutoff: float = 1.0):
        self.filter_x = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self.filter_y = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self.filter_z = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()

    def filter(self, x: float, y: float, z: float, timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        if timestamp is None:
            timestamp = time.time()
        return (
            self.filter_x.filter(x, timestamp),
            self.filter_y.filter(y, timestamp),
            self.filter_z.filter(z, timestamp),
        )


class OneEuroQuaternionFilter:
    """
    Adaptive One-Euro Filter for 3D unit quaternions (w, x, y, z) using Slerp.
    Computes instantaneous angular velocity (rad/s) to scale dynamic cutoff frequency,
    eliminating hand jitter at rest while preserving instant responsiveness for wrist swings.
    """
    def __init__(self, freq: float = 60.0, mincutoff: float = 1.0, beta: float = 2.0, dcutoff: float = 1.0):
        self.freq = max(1.0, freq)
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.omega_filter = LowPassFilter()
        self.last_time: Optional[float] = None
        self.last_raw_q: Optional[Quaternion] = None
        self.q_filtered: Optional[Quaternion] = None

    def reset(self):
        self.omega_filter.reset()
        self.last_time = None
        self.last_raw_q = None
        self.q_filtered = None

    def filter(self, q: Quaternion, timestamp: Optional[float] = None) -> Quaternion:
        if timestamp is None:
            timestamp = time.time()

        q_curr = quat_normalize(q)

        if self.last_time is None or self.q_filtered is None or self.last_raw_q is None:
            self.last_time = timestamp
            self.last_raw_q = q_curr
            self.q_filtered = q_curr
            return q_curr

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 1e-5:
            dt = 1.0 / self.freq
        elif dt > 0.4:
            self.reset()
            self.last_time = timestamp
            self.last_raw_q = q_curr
            self.q_filtered = q_curr
            return q_curr

        # Keep raw quaternion in same hemisphere as previous sample before computing angular speed
        dot_raw = abs(quat_dot(self.last_raw_q, q_curr))
        dot_raw = max(0.0, min(1.0, dot_raw))
        delta_angle = 2.0 * math.acos(dot_raw)
        angular_speed = delta_angle / dt
        self.last_raw_q = q_curr

        # Smooth angular velocity magnitude to stabilize cutoff transitions
        a_d = _one_euro_alpha(self.dcutoff, dt)
        filtered_omega = self.omega_filter.filter(angular_speed, a_d)

        cutoff = self.mincutoff + self.beta * abs(filtered_omega)
        alpha = _one_euro_alpha(cutoff, dt)

        self.q_filtered = quat_slerp(self.q_filtered, q_curr, alpha)
        return self.q_filtered


class AdaptivePointerPrecision:
    """
    Speed-adaptive sensitivity curve tailored for human hand control:
    - Micro-tremor (< 1.8px): locks cursor completely still.
    - Fine precision zone (1.8px to 7px): gentle 0.55x - 0.85x damping for easy targeting of small buttons/links.
    - Normal movement (7px to 22px): natural 1:1 linear tracking.
    - Fast sweeps (> 22px): smooth acceleration (up to 1.35x) to easily span the full monitor without arm strain.
    """
    def __init__(self, tremor_threshold: float = 1.8):
        self.tremor_threshold = tremor_threshold
        self.last_output: Optional[Tuple[float, float]] = None

    def apply(self, target_x: float, target_y: float) -> Tuple[float, float]:
        if self.last_output is None:
            self.last_output = (target_x, target_y)
            return target_x, target_y

        dx = target_x - self.last_output[0]
        dy = target_y - self.last_output[1]
        dist = math.hypot(dx, dy)

        # Micro-tremor suppression: lock cursor still
        if dist < self.tremor_threshold:
            return self.last_output

        # Precision curve multiplier based on speed
        if dist < 7.0:
            # Slow precision zone: damp by 0.55x
            gain = 0.55 + 0.45 * ((dist - self.tremor_threshold) / (7.0 - self.tremor_threshold))
        elif dist < 22.0:
            # Cruising zone: smooth transition from 1.0x to 1.15x
            gain = 1.0 + 0.15 * ((dist - 7.0) / 15.0)
        else:
            # Fast sweep zone: acceleration up to 1.35x
            gain = min(1.35, 1.15 + 0.20 * min(1.0, (dist - 22.0) / 30.0))

        new_x = self.last_output[0] + dx * gain
        new_y = self.last_output[1] + dy * gain
        self.last_output = (new_x, new_y)
        return new_x, new_y

    def reset(self):
        self.last_output = None


class DeadZoneMapper:
    """Sub-pixel deadzone mapper for backward compatibility."""
    def __init__(self, deadzone_radius: float = 1.5, sensitivity: float = 1.0):
        self.precision_curve = AdaptivePointerPrecision(tremor_threshold=deadzone_radius)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        return self.precision_curve.apply(x, y)

    def reset(self):
        self.precision_curve.reset()
