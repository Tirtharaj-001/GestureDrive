"""
vr - GestureDrive Virtual Reality & 6-DOF Spatial Subsystem.
Provides 3D metric spatial transformation, OpenXR/SteamVR coordinate mapping,
VRChat OSC finger flexion streaming, and SteamVR VMT virtual controller emulation.
"""

from vr.spatial import (
    MetricLandmarkPoint,
    CameraMountMode,
    PinholeCameraModel,
    DepthEstimator,
    SteamVRCoordinateMapper,
    VRPlayspaceCalibrator,
)
from vr.pose6dof import (
    Hand6DOFTracker,
    compute_palm_orientation_quaternion,
    quaternion_to_euler_deg,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    compute_angular_velocity,
)
from vr.osc_vrchat import (
    VRChatGesture,
    VRChatOSCClient,
    compute_finger_curls_and_spreads,
    map_hand_to_vrchat_gesture,
)
from vr.vmt_client import (
    VMTClient,
    VMTMode,
    VMTDeviceRole,
)
from vr.vmt_driver import (
    check_vmt_driver_status,
    register_driver_in_steamvr,
    unregister_driver_in_steamvr,
)

__all__ = [
    "MetricLandmarkPoint",
    "CameraMountMode",
    "PinholeCameraModel",
    "DepthEstimator",
    "SteamVRCoordinateMapper",
    "VRPlayspaceCalibrator",
    "Hand6DOFTracker",
    "compute_palm_orientation_quaternion",
    "quaternion_to_euler_deg",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "compute_angular_velocity",
    "VRChatGesture",
    "VRChatOSCClient",
    "compute_finger_curls_and_spreads",
    "map_hand_to_vrchat_gesture",
    "VMTClient",
    "VMTMode",
    "VMTDeviceRole",
    "check_vmt_driver_status",
    "register_driver_in_steamvr",
    "unregister_driver_in_steamvr",
]
