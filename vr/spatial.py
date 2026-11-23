"""
vr/spatial.py - 3D Metric Depth Estimation & VR Coordinate Alignment.
Transforms 2D camera landmarks and MediaPipe relative depth into calibrated real-world
metric space (X, Y, Z in meters) aligned with OpenXR / SteamVR playspace standards.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from vision.landmarks import (
    LandmarkPoint, WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP,
    INDEX_TIP, THUMB_TIP, dist_2d, dist_3d, get_palm_center
)
from vr.pose6dof import Hand6DOFTracker


class CameraMountMode(Enum):
    """Camera positioning relative to the user."""
    DESK_FACING_USER = "Desk (Facing User)"     # Standard webcam looking at user from monitor
    HMD_MOUNTED = "HMD Mounted (Forward)"       # Camera mounted on front of VR headset looking outward


class MetricLandmarkPoint:
    """Represents a hand landmark in real-world 3D metric coordinates (meters)."""
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)  # meters
        self.y = float(y)  # meters
        self.z = float(z)  # meters

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"MetricPoint(x={self.x:.3f}m, y={self.y:.3f}m, z={self.z:.3f}m)"


class PinholeCameraModel:
    """
    Standard pinhole camera projection model.
    Computes intrinsic focal lengths and converts pixel coordinates to directional 3D rays.
    """
    def __init__(self, width: int = 640, height: int = 480, hfov_deg: float = 65.0):
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg

        # Compute focal length in pixels from horizontal FOV
        hfov_rad = math.radians(hfov_deg)
        self.fx = (self.width / 2.0) / math.tan(hfov_rad / 2.0)

        # Aspect ratio preserves square pixels
        vfov_rad = 2.0 * math.atan((self.height / self.width) * math.tan(hfov_rad / 2.0))
        self.vfov_deg = math.degrees(vfov_rad)
        self.fy = (self.height / 2.0) / math.tan(vfov_rad / 2.0)

        # Optical center (principal point)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def pixel_to_ray(self, px: float, py: float) -> Tuple[float, float, float]:
        """
        Converts pixel coordinate (px, py) to a normalized 3D ray direction vector in camera space.
        """
        norm_x = (px - self.cx) / self.fx
        norm_y = (py - self.cy) / self.fy
        norm_z = 1.0
        length = math.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z)
        return (norm_x / length, norm_y / length, norm_z / length)


class DepthEstimator:
    """
    Estimates true physical metric depth (meters) from apparent hand pixel dimensions.
    Based on anthropometric adult hand biomechanics:
    Average adult Wrist-to-Middle-MCP distance is ~9.5 cm (0.095m).
    """
    # Average physical bone distances in adult humans (in meters)
    DEFAULT_WRIST_TO_MIDDLE_MCP_M: float = 0.095  # 9.5 cm
    DEFAULT_PALM_WIDTH_M: float = 0.084           # 8.4 cm (Index MCP to Pinky MCP)

    def __init__(self,
                 camera_model: PinholeCameraModel,
                 physical_hand_scale_m: float = DEFAULT_WRIST_TO_MIDDLE_MCP_M):
        self.cam = camera_model
        self.physical_hand_scale_m = physical_hand_scale_m

    def estimate_wrist_depth(self, landmarks: List[LandmarkPoint]) -> float:
        """
        Calculates distance Z (in meters) from camera to wrist using apparent pixel scale.
        Z = (focal_length * real_bone_length) / apparent_bone_pixels
        """
        if not landmarks or len(landmarks) < 21:
            return 0.65  # Default resting distance ~65cm

        w = landmarks[WRIST]
        m = landmarks[MIDDLE_MCP]

        # Calculate pixel distance between Wrist and Middle MCP
        dx_px = (m.x - w.x) * self.cam.width
        dy_px = (m.y - w.y) * self.cam.height
        bone_px = math.hypot(dx_px, dy_px)

        if bone_px < 5.0:
            return 0.65

        # Perspective depth formula
        z_meters = (self.cam.fx * self.physical_hand_scale_m) / bone_px

        # Clamp to realistic desktop/VR interaction reach [0.20m to 1.80m]
        return max(0.20, min(1.80, z_meters))

    def convert_to_metric_skeleton(self,
                                  landmarks: List[LandmarkPoint]) -> Tuple[List[MetricLandmarkPoint], float]:
        """
        Converts all 21 normalized landmarks into metric 3D points (meters) relative to camera.
        Returns (list_of_metric_landmarks, estimated_wrist_depth_m).
        """
        wrist_z_m = self.estimate_wrist_depth(landmarks)
        metric_points = []

        for lm in landmarks:
            # Reconstruct X and Y in camera space from pinhole ray at depth Z
            px = lm.x * self.cam.width
            py = lm.y * self.cam.height

            # MediaPipe provides landmark 'z' relative to wrist on roughly the same scale as landmark x
            # Scale relative z to meters proportional to physical hand size
            relative_z_m = lm.z * (self.physical_hand_scale_m * 1.5)
            point_z_m = wrist_z_m + relative_z_m

            # Pinhole back-projection: X = (px - cx) * Z / fx
            point_x_m = (px - self.cam.cx) * point_z_m / self.cam.fx
            point_y_m = (py - self.cam.cy) * point_z_m / self.cam.fy

            metric_points.append(MetricLandmarkPoint(point_x_m, point_y_m, point_z_m))

        return metric_points, wrist_z_m


class SteamVRCoordinateMapper:
    """
    Transforms camera-space 3D coordinates into OpenXR / SteamVR playspace coordinates.

    OpenCV Camera Frame:
      +X: Right
      +Y: Down
      +Z: Forward (away from camera lens)

    SteamVR / OpenXR World Frame:
      +X: Right (user's right)
      +Y: Up (towards ceiling)
      +Z: Backward (towards user in OpenXR) or Forward into VR space
    """
    def __init__(self, mount_mode: CameraMountMode = CameraMountMode.DESK_FACING_USER):
        self.mount_mode = mount_mode

    def to_vr_space(self,
                    metric_point: MetricLandmarkPoint,
                    origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> MetricLandmarkPoint:
        """
        Converts a camera-space MetricLandmarkPoint into SteamVR/OpenXR playspace coordinates (meters),
        subtracting a calibrated user center origin.
        """
        cx, cy, cz = metric_point.x, metric_point.y, metric_point.z
        ox, oy, oz = origin_offset

        if self.mount_mode == CameraMountMode.DESK_FACING_USER:
            # Desk webcam looks towards the user:
            # - Reaching right in front of mirror view = +X in VR
            # - Reaching UP = -Y in camera space = +Y in VR
            # - Reaching forward towards camera = decreasing camera depth = +Z in VR (extending arm into VR world)
            vr_x = (cx - ox)
            vr_y = -(cy - oy)
            vr_z = -(cz - oz)  # Moving closer to camera means extending arms into VR space
        else:
            # HMD mounted camera looks forward into room:
            # - Hand to right = +X
            # - Hand up = -Y in camera = +Y in VR
            # - Hand forward = +Z in camera = +Z in VR
            vr_x = (cx - ox)
            vr_y = -(cy - oy)
            vr_z = (cz - oz)

        return MetricLandmarkPoint(vr_x, vr_y, vr_z)


class VRPlayspaceCalibrator:
    """
    Calibrates user center origin (0, 0, 0) in metric space so hands naturally align with
    VR chest / controller rest positions, and runs 6-DOF smoothing + velocity estimation.
    """
    def __init__(self,
                 mount_mode: CameraMountMode = CameraMountMode.DESK_FACING_USER,
                 camera_width: int = 640,
                 camera_height: int = 480):
        self.cam_model = PinholeCameraModel(width=camera_width, height=camera_height)
        self.depth_estimator = DepthEstimator(self.cam_model)
        self.mapper = SteamVRCoordinateMapper(mount_mode)
        self.pose_tracker = Hand6DOFTracker(freq=60.0)

        # Calibrated resting origin (X, Y, Z in camera metric space)
        # Default: centered X=0m, chest height Y=-0.1m, distance Z=0.60m
        self.origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.60)
        self.is_calibrated: bool = False

    def update_camera_dimensions(self, camera_width: int, camera_height: int):
        """Updates intrinsic camera dimensions without resetting origin or filter memory."""
        if self.cam_model.width != camera_width or self.cam_model.height != camera_height:
            self.cam_model = PinholeCameraModel(width=camera_width, height=camera_height)
            self.depth_estimator = DepthEstimator(self.cam_model)

    def reset(self):
        """Resets playspace origin and 6-DOF smoothing/velocity filters."""
        self.origin_offset = (0.0, 0.0, 0.60)
        self.is_calibrated = False
        self.pose_tracker.reset()

    def calibrate_origin(self, primary_hand_landmarks: List[LandmarkPoint]):
        """
        Recalibrates (0, 0, 0) origin to current hand position.
        """
        metric_points, wrist_depth = self.depth_estimator.convert_to_metric_skeleton(primary_hand_landmarks)
        if metric_points:
            palm = metric_points[0]  # Wrist point
            self.origin_offset = (palm.x, palm.y, wrist_depth)
            self.is_calibrated = True
            # Reset filters so sudden origin shift does not create artificial velocity spike
            self.pose_tracker.reset()
            print(f"[VR Playspace] Recalibrated VR Origin to: X={palm.x:+.3f}m, Y={palm.y:+.3f}m, Z={wrist_depth:.3f}m")

    def prune_stale_hands(self, active_keys: set, timestamp: Optional[float] = None):
        """Clears 6-DOF filter history for hands that left the camera view."""
        self.pose_tracker.prune_stale_hands(active_keys, timestamp=timestamp)

    def process_hand_to_vr(
        self,
        landmarks: List[LandmarkPoint],
        handedness: str = "Right",
        timestamp: Optional[float] = None,
        track_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full 6-DOF VR spatial pipeline:
        Normalized Landmarks -> Metric 3D Skeleton -> SteamVR Playspace Coordinates
        -> 3D Position & Slerp Quaternion One-Euro Filtering + Linear/Angular Velocity.
        """
        metric_skeleton, wrist_depth = self.depth_estimator.convert_to_metric_skeleton(landmarks)

        # Convert every joint to VR space
        vr_skeleton = [self.mapper.to_vr_space(pt, self.origin_offset) for pt in metric_skeleton]

        vr_wrist = vr_skeleton[WRIST]
        vr_index_tip = vr_skeleton[INDEX_TIP]

        # Calculate raw VR palm center
        w = vr_skeleton[WRIST]
        i = vr_skeleton[INDEX_MCP]
        p = vr_skeleton[PINKY_MCP]
        vr_palm_raw = MetricLandmarkPoint((w.x + i.x + p.x) / 3.0, (w.y + i.y + p.y) / 3.0, (w.z + i.z + p.z) / 3.0)

        # Smooth 3D position + palm orientation quaternion + linear/angular velocities
        pose_6dof = self.pose_tracker.update_hand(
            vr_palm_raw.to_tuple(),
            vr_skeleton,
            handedness=handedness,
            timestamp=timestamp,
            track_key=track_key,
        )
        sx, sy, sz = pose_6dof["position_m"]
        vr_palm = MetricLandmarkPoint(sx, sy, sz)

        # Compute hand reach distance from calibrated center using smoothed position
        reach_distance_m = math.sqrt(vr_palm.x**2 + vr_palm.y**2 + vr_palm.z**2)

        return {
            "metric_skeleton": metric_skeleton,
            "vr_skeleton": vr_skeleton,
            "wrist_depth_m": wrist_depth,
            "vr_wrist": vr_wrist,
            "vr_palm": vr_palm,
            "vr_palm_raw": vr_palm_raw,
            "vr_index_tip": vr_index_tip,
            "reach_distance_m": reach_distance_m,
            "origin_offset": self.origin_offset,
            "quaternion": pose_6dof["quaternion"],
            "raw_quaternion": pose_6dof["raw_quaternion"],
            "euler_deg": pose_6dof["euler_deg"],
            "basis_axes": pose_6dof["basis_axes"],
            "linear_velocity": pose_6dof["linear_velocity"],
            "linear_speed": pose_6dof["linear_speed"],
            "angular_velocity": pose_6dof["angular_velocity"],
            "angular_speed_rad": pose_6dof["angular_speed_rad"],
            "angular_speed_deg": pose_6dof["angular_speed_deg"],
            "track_key": pose_6dof["track_key"],
        }

