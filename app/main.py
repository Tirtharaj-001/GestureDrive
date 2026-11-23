"""
app/main.py - Dual-Mode GestureDrive Engine.
- BASIC MODE: Desktop PC Mouse Controller, One-Euro Smoothing, Workspace Calibration, Clicks & Drag.
- ADVANCED VR MODE: 3D Metric Depth Estimation, OpenXR/SteamVR Playspace Alignment & 6-DOF Telemetry.
Toggle seamlessly between Basic and Advanced VR modes using the [V] key.
Includes background throttling prevention, zero-wait frame sync, and non-blocking pollKey when minimized.
"""

import sys
import os
import time
from enum import Enum
import cv2
import numpy as np

# Disable bytecode (.pyc) generation
sys.dont_write_bytecode = True

# Add project root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.camera import CameraStream
from vision.hand_tracker import HandTracker
from vision.landmarks import (
    INDEX_TIP, INDEX_DIP, INDEX_MCP, THUMB_TIP, WRIST,
    get_finger_states, get_hand_scale, get_palm_center
)
from vision.filters import OneEuroFilter2D, AdaptivePointerPrecision
from vision.calibration import HandCalibrator
from controller.mouse_controller import MouseController, MouseActionState
from vr.spatial import VRPlayspaceCalibrator, CameraMountMode
from vr.osc_vrchat import VRChatOSCClient
from vr.vmt_client import VMTClient, VMTMode
from vr.desktop_gestures import VRChatDesktopGestureEmulator


class AppMode(Enum):
    BASIC = "Basic Mode (Desktop PC Controller)"
    ADVANCED_VR = "Advanced VR Mode (SteamVR 3D Playspace)"


def configure_windows_background_performance():
    """
    Prevents Windows from throttling process priority and timers when window is minimized or in background.
    """
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes
        # Request 1ms timer precision to prevent background timer degradation
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod.argtypes = [wintypes.UINT]
        winmm.timeBeginPeriod(1)

        # Set process priority to ABOVE_NORMAL_PRIORITY_CLASS (0x8000)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x8000)

        # Prevent thread sleep and EcoQoS background throttling
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

        # User32 for checking minimized (iconic) window state
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL

        return user32, winmm
    except Exception as e:
        print(f"[System] Notice: Windows background performance tuning: {e}")
        return None, None


def draw_cyber_box(frame, x1, y1, x2, y2, color=(0, 255, 230), corner_len=18, thickness=1):
    """Draws an active workspace box with tactical corner brackets."""
    sub_img = frame[y1:y2, x1:x2]
    if sub_img.size > 0:
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 20
        cv2.addWeighted(sub_img, 0.92, white_rect, 0.08, 0, sub_img)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 80), thickness, cv2.LINE_AA)

    # Corners
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 2, cv2.LINE_AA)

    cv2.putText(frame, "ACTIVE WORKSPACE", (x1 + 6, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_crosshair(frame, px, py, color=(0, 255, 255), size=12):
    """Draws a targeting reticle at the interaction cursor point."""
    cv2.circle(frame, (px, py), size, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)
    cv2.line(frame, (px - size - 4, py), (px - 4, py), color, 1, cv2.LINE_AA)
    cv2.line(frame, (px + 4, py), (px + size + 4, py), color, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py - size - 4), (px, py - 4), color, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py + 4), (px, py + size + 4), color, 1, cv2.LINE_AA)


def draw_vr_depth_meter(frame, x, y, w, h, depth_m, min_d=0.25, max_d=1.20):
    """Renders a 3D metric depth gauge bar on the HUD."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 40), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 1)

    norm = max(0.0, min(1.0, (depth_m - min_d) / (max_d - min_d)))
    fill_h = int(norm * (h - 4))
    fill_y = y + h - 2 - fill_h

    bar_color = (0, 255, 0) if depth_m < 0.75 else ((0, 255, 255) if depth_m < 1.0 else (0, 100, 255))
    cv2.rectangle(frame, (x + 2, fill_y), (x + w - 2, y + h - 2), bar_color, cv2.FILLED)
    cv2.putText(frame, f"Z: {depth_m:.2f}m", (x - 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


def draw_vr_orientation_gizmo(frame, px, py, basis_axes, axis_len=42):
    """Projects 3D local hand axes (X=Right, Y=Up/Back, Z=Forward) onto the camera HUD."""
    x_axis, y_axis, z_axis = basis_axes
    axes = [
        (x_axis, (0, 0, 255), "X"),     # Red: Right across palm
        (y_axis, (0, 255, 0), "Y"),     # Green: Out back of palm
        (z_axis, (255, 220, 0), "Z"),   # Cyan: Forward along fingers
    ]
    for vec, color, label in axes:
        # Image +X matches VR +X; image +Y goes down while VR +Y goes up
        end_x = int(px + vec[0] * axis_len)
        end_y = int(py - vec[1] * axis_len)
        cv2.arrowedLine(frame, (px, py), (end_x, end_y), color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(frame, label, (end_x + 3, end_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def main():
    print("=================================================================")
    print("  GESTUREDRIVE — Dual-Mode: Basic PC Mouse & Advanced VR 3D Core")
    print("=================================================================")
    print("  Mode Switching:")
    print("    [V]    : Toggle BASIC MODE  <==>  ADVANCED VR MODE")
    print("  Basic Mode Controls:")
    print("    [M]    : Toggle Mouse Control ON / OFF (Clutch)")
    print("    [C]    : Calibrate Desktop Workspace")
    print("    [+/-]  : Increase / Decrease Motion Scale & Sensitivity")
    print("    [[ / ]]: Expand / Narrow Active Workspace Area")
    print("  Advanced VR Mode Controls:")
    print("    [C]    : Zero & Recalibrate VR Playspace Origin (0, 0, 0)")
    print("    [H]    : Toggle Camera Mount (Desk-Facing <==> HMD-Mounted)")
    print("    [O]    : Toggle VRChat OSC Stream (127.0.0.1:9000)")
    print("    [T]    : Toggle SteamVR VMT Stream (127.0.0.1:39570)")
    print("    [Y]    : Cycle VMT Mode (Controllers <==> Trackers)")
    print("  General Controls:")
    print("    [R]    : Reset All Filters & Calibration")
    print("    [Q]    : Exit GestureDrive")
    print("=================================================================\n")

    # Configure Windows to prevent background throttling when minimized
    user32, winmm = configure_windows_background_performance()
    window_name = "GestureDrive - Dual-Mode (Basic & VR)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    hwnd = None

    # Initialize Threaded Camera Stream (1280x720 Widescreen)
    camera = CameraStream(width=1280, height=720).start()

    # Initialize MediaPipe Hand Tracker in VIDEO mode with edge sensitivity
    tracker = HandTracker(max_hands=2, min_detection_con=0.45, min_track_con=0.45)

    # Initialize Basic Mode Components (Zero-Lag One-Euro Smoothing & Wide Active Workspace)
    cursor_filter = OneEuroFilter2D(freq=60.0, mincutoff=0.5, beta=0.030, dcutoff=1.0)
    precision_curve = AdaptivePointerPrecision(tremor_threshold=1.8)
    calibrator = HandCalibrator(margin_x=0.08, margin_y=0.08, motion_scale=1.20, sampling_frames=30)
    mouse = MouseController(click_debounce_sec=0.28, drag_hold_threshold_sec=0.22)

    # Initialize Advanced VR Spatial & VRChat OSC Components
    vr_calibrator = VRPlayspaceCalibrator(
        mount_mode=CameraMountMode.DESK_FACING_USER,
        camera_width=1280,
        camera_height=720
    )
    vrc_osc = VRChatOSCClient(host="127.0.0.1", port=9000)
    vmt = VMTClient(host="127.0.0.1", port=39570, mode=VMTMode.CONTROLLERS, target_hz=60.0)
    desktop_gestures = VRChatDesktopGestureEmulator(enabled=True)

    # Current Application Mode (supports --vr CLI argument)
    start_in_vr = "--vr" in sys.argv
    mode = AppMode.ADVANCED_VR if start_in_vr else AppMode.BASIC
    if start_in_vr:
        vr_calibrator.pose_tracker.reset()
        if vmt.enabled:
            vmt.spawn_devices()
        print("\n>>> LAUNCHED DIRECTLY IN: ADVANCED VR MODE (Streaming to VRChat on 127.0.0.1:9000)\n")

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    last_fid = -1

    # Out-of-frame tracking retention buffer
    hand_lost_time = None
    last_valid_screen_pos = None
    GRACE_PERIOD_SEC = 0.35

    try:
        while True:
            # Non-blocking lock-free check for fresh camera frames (1ms poll)
            grabbed, frame, fid = camera.read_latest()
            if not grabbed or frame is None or fid == last_fid:
                time.sleep(0.001)
                continue
            last_fid = fid

            # Check if camera window is minimized
            is_minimized = False
            if user32:
                if hwnd is None:
                    hwnd = user32.FindWindowW(None, window_name)
                if hwnd:
                    is_minimized = bool(user32.IsIconic(hwnd))

            # Mirror frame for desk/selfie view; keep HMD forward-facing view unmirrored
            is_mirrored = not (
                mode == AppMode.ADVANCED_VR
                and vr_calibrator.mapper.mount_mode == CameraMountMode.HMD_MOUNTED
            )
            if is_mirrored:
                frame = cv2.flip(frame, 1)
            h_cam, w_cam, _ = frame.shape

            # Dynamically adapt camera dimensions if changed
            vr_calibrator.update_camera_dimensions(w_cam, h_cam)

            # FPS calculation
            frame_count += 1
            now = time.time()
            dt = now - fps_timer
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_timer = now

            # Downsample for ultra-fast MediaPipe inference (saves 30% CPU, identical normalized output)
            if w_cam > 640:
                inf_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            else:
                inf_frame = frame

            # Process hand landmarks via MediaPipe Tasks API (invert Left/Right when mirrored)
            hands = tracker.process_frame(inf_frame, mirrored=is_mirrored)

            # Re-scale pixel coordinates to full frame for drawing
            scale_x = w_cam / float(inf_frame.shape[1])
            scale_y = h_cam / float(inf_frame.shape[0])
            for h in hands:
                h["pixel_coords"] = [(int(px * scale_x), int(py * scale_y)) for (px, py) in h["pixel_coords"]]
                ppx, ppy = h["palm_pixel"]
                h["palm_pixel"] = (int(ppx * scale_x), int(ppy * scale_y))

            # Only draw cyber skeleton when window is visible (skips drawing when minimized for max speed)
            if not is_minimized:
                tracker.draw_hands(frame, hands)

            # Primary hand selection (prefer Pointing/Pinch or Right hand)
            primary_hand = None
            if hands:
                for h in hands:
                    g = h.get("gesture_name", "")
                    if g in ("Pointing", "Pinch", "OK"):
                        primary_hand = h
                        break
                if primary_hand is None:
                    primary_hand = next((h for h in hands if h["handedness"] == "Right"), hands[0])

            # Basic Mode: Desktop PC Controller
            if mode == AppMode.BASIC:
                wx1, wy1, wx2, wy2 = calibrator.get_workspace_rect(w_cam, h_cam)
                if not is_minimized:
                    draw_cyber_box(frame, wx1, wy1, wx2, wy2, color=(0, 255, 230))

                # Handle Calibration if active
                calibrating_now = calibrator.is_calibrating
                if calibrating_now and primary_hand:
                    calibrator.update_calibration(primary_hand["landmarks"])

                smoothed_cam_x, smoothed_cam_y = None, None
                target_screen_x, target_screen_y = None, None

                if primary_hand:
                    hand_lost_time = None
                    landmarks = primary_hand["landmarks"]
                    gesture_name = primary_hand.get("gesture_name", "None")

                    if gesture_name in ("Pointing", "Pinch", "OK"):
                        # Blend 70% fingertip with 30% DIP joint to suppress anatomical fingertip tremor
                        tip = landmarks[INDEX_TIP]
                        dip = landmarks[INDEX_DIP]
                        px = tip.x * 0.70 + dip.x * 0.30
                        py = tip.y * 0.70 + dip.y * 0.30
                    else:
                        palm = primary_hand["palm_center"]
                        px, py = palm.x, palm.y

                    raw_screen_x, raw_screen_y = calibrator.map_camera_to_screen(
                        px, py, mouse.screen_width, mouse.screen_height
                    )
                    smooth_scr_x, smooth_scr_y = cursor_filter.filter(raw_screen_x, raw_screen_y, timestamp=now)
                    final_scr_x, final_scr_y = precision_curve.apply(smooth_scr_x, smooth_scr_y)
                    target_screen_x, target_screen_y = int(final_scr_x), int(final_scr_y)
                    last_valid_screen_pos = (target_screen_x, target_screen_y)

                    norm_back_x = final_scr_x / max(1, mouse.screen_width)
                    norm_back_y = final_scr_y / max(1, mouse.screen_height)
                    smoothed_cam_x = max(0, min(w_cam - 1, int(wx1 + norm_back_x * (wx2 - wx1))))
                    smoothed_cam_y = max(0, min(h_cam - 1, int(wy1 + norm_back_y * (wy2 - wy1))))

                    mouse.update(gesture_name, (target_screen_x, target_screen_y), hand_visible=True)

                    if not is_minimized:
                        reticle_color = (0, 255, 0)
                        if mouse.state == MouseActionState.DRAGGING:
                            reticle_color = (0, 0, 255)
                        elif mouse.state == MouseActionState.CLICKING:
                            reticle_color = (255, 0, 255)
                        elif mouse.state == MouseActionState.RIGHT_CLICK:
                            reticle_color = (255, 200, 0)
                        elif mouse.state == MouseActionState.PAUSED:
                            reticle_color = (120, 120, 120)

                        draw_crosshair(frame, smoothed_cam_x, smoothed_cam_y, color=reticle_color, size=14)

                else:
                    # Hand momentarily out of camera: apply grace period to prevent snapping
                    if last_valid_screen_pos is not None:
                        if hand_lost_time is None:
                            hand_lost_time = now
                        elapsed_lost = now - hand_lost_time
                        if elapsed_lost < GRACE_PERIOD_SEC:
                            mouse.move_to(last_valid_screen_pos[0], last_valid_screen_pos[1])
                        else:
                            mouse.update("None", (0, 0), hand_visible=False)
                            cursor_filter.reset()
                            precision_curve.reset()
                            last_valid_screen_pos = None
                            hand_lost_time = None
                    else:
                        mouse.update("None", (0, 0), hand_visible=False)
                        cursor_filter.reset()
                        precision_curve.reset()

                # Basic Mode Telemetry Badges (rendered only when visible)
                if not is_minimized:
                    for hand in hands:
                        hand_id = hand["id"]
                        handedness = hand["handedness"]
                        conf = hand["confidence"]
                        px, py = hand["palm_pixel"]
                        landmarks = hand["landmarks"]
                        gesture_name = hand.get("gesture_name", "None")

                        f_states = get_finger_states(landmarks, handedness)
                        ext_count = sum(1 for v in f_states.values() if v)

                        badge_w, badge_h = 175, 82
                        bx = max(10, min(w_cam - badge_w - 10, px - 85))
                        by = max(50, min(h_cam - badge_h - 10, py - 95))

                        border_color = (0, 255, 200)
                        if gesture_name == "Open Palm":
                            border_color = (0, 255, 0)
                        elif gesture_name == "Fist":
                            border_color = (0, 0, 255)
                        elif gesture_name in ("Pinch", "OK"):
                            border_color = (255, 0, 255)
                        elif gesture_name == "Pointing":
                            border_color = (255, 255, 0)

                        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (20, 20, 30), cv2.FILLED)
                        cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), border_color, 1)

                        cv2.putText(frame, f"Hand: {handedness} ({conf*100:.0f}%)", (bx + 10, by + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Gesture: {gesture_name}", (bx + 10, by + 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border_color, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Fingers Open: {ext_count}/5", (bx + 10, by + 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Palm Pos: ({px}, {py})", (bx + 10, by + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

            # Advanced VR Mode: SteamVR 3D Playspace, 6-DOF Quaternions & Velocities
            else:
                if mouse.is_left_down:
                    mouse.left_up()

                if len(hands) == 0:
                    desktop_gestures.release_all()

                seen_left = False
                seen_right = False
                active_vr_keys = set()
                for idx_h, hand in enumerate(hands):
                    landmarks = hand["landmarks"]
                    handedness = hand["handedness"]
                    px, py = hand["palm_pixel"]
                    gesture_name = hand.get("gesture_name", "None")

                    # Emulate native VRChat desktop gesture hotkeys (Shift + F1..F8)
                    desktop_gestures.update_hand_gesture(handedness, gesture_name)

                    # Tie-break track key if detector momentarily reports duplicate handedness in one frame
                    tkey = handedness if handedness not in active_vr_keys else f"{handedness}_{idx_h}"
                    active_vr_keys.add(tkey)

                    vr_data = vr_calibrator.process_hand_to_vr(
                        landmarks, handedness=handedness, timestamp=now, track_key=tkey
                    )
                    osc_data = vrc_osc.send_hand_state(
                        handedness, vr_data["metric_skeleton"], gesture_name=gesture_name, timestamp=now
                    )
                    vmt_data = vmt.send_hand_pose(
                        handedness=handedness,
                        position=(vr_data["vr_palm"].x, vr_data["vr_palm"].y, vr_data["vr_palm"].z),
                        quaternion=vr_data["quaternion"],
                        euler_deg=vr_data["euler_deg"],
                        gesture_name=gesture_name,
                        landmarks=vr_data["vr_skeleton"],
                        curls=osc_data.get("curls"),
                        timestamp=now,
                    )

                    # Stream locomotion & rotation to VRChat OSC & Desktop Keyboard (WASD & Arrows):
                    # - Left Hand tilt: Walk & Strafe (/input/Vertical, /input/Horizontal & W/A/S/D)
                    # - Right Hand tilt: Rotate & Turn view (/input/LookHorizontal & Left/Right Arrows)
                    if handedness == "Left":
                        seen_left = True
                        vrc_osc.send_locomotion(vmt_data["joy_x"], vmt_data["joy_y"])
                        desktop_gestures.update_locomotion(vmt_data["joy_x"], vmt_data["joy_y"])
                    elif handedness == "Right":
                        seen_right = True
                        vrc_osc.send_look(vmt_data["joy_x"])
                        desktop_gestures.update_look(vmt_data["joy_x"])

                    wrist_depth = vr_data["wrist_depth_m"]
                    vr_palm = vr_data["vr_palm"]
                    quat = vr_data["quaternion"]
                    pitch, yaw, roll = vr_data["euler_deg"]
                    lin_speed = vr_data["linear_speed"]
                    ang_speed_deg = vr_data["angular_speed_deg"]

                    if not is_minimized:
                        draw_vr_orientation_gizmo(frame, px, py, vr_data["basis_axes"], axis_len=42)
                        draw_vr_depth_meter(frame, px + 80, py - 45, 12, 70, wrist_depth)

                        vbadge_w, vbadge_h = 280, 176
                        bx = max(10, min(w_cam - vbadge_w - 10, px - 140))
                        by = max(48, min(h_cam - vbadge_h - 10, py - 180))

                        cv2.rectangle(frame, (bx, by), (bx + vbadge_w, by + vbadge_h), (25, 20, 35), cv2.FILLED)
                        cv2.rectangle(frame, (bx, by), (bx + vbadge_w, by + vbadge_h), (255, 0, 180), 1)

                        cv2.putText(frame, f"VR 6-DOF: {handedness} [{gesture_name}]", (bx + 8, by + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Pos: ({vr_palm.x:+.2f}, {vr_palm.y:+.2f}, {vr_palm.z:+.2f})m", (bx + 8, by + 36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 230), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Rot P/Y/R: {pitch:+.0f} / {yaw:+.0f} / {roll:+.0f} deg", (bx + 8, by + 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 200, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Quat: ({quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f})", (bx + 8, by + 72),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 180, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Vel: {lin_speed:.2f} m/s | {ang_speed_deg:.0f} deg/s", (bx + 8, by + 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (120, 255, 140), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"VRC: {osc_data['gesture_name']} ({osc_data['gesture_code']}) | W:{osc_data['weight']:.2f}", (bx + 8, by + 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 220, 255), 1, cv2.LINE_AA)
                        curls = osc_data["curls"]
                        curl_str = f"Curl T/I/M/R/P: {curls['Thumb']:.1f} {curls['Index']:.1f} {curls['Middle']:.1f} {curls['Ring']:.1f} {curls['Pinky']:.1f}"
                        cv2.putText(frame, curl_str, (bx + 8, by + 126),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 190, 200), 1, cv2.LINE_AA)
                        role_tag = "L_CTRL" if vmt_data["index"] == 0 else "R_CTRL"
                        if vmt.mode == VMTMode.TRACKERS:
                            role_tag = f"TRK_{vmt_data['index']}"
                        btn_label = "X/Y" if vmt_data["index"] == 0 else "A/B"
                        vmt_line1 = f"VMT: {role_tag} | Trig:{vmt_data['trigger']:.2f} Grip:{vmt_data['grip']:.2f} {btn_label}:[{vmt_data['button_a']},{vmt_data['button_b']}]"
                        cv2.putText(frame, vmt_line1, (bx + 8, by + 146),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 180), 1, cv2.LINE_AA)
                        walk_state = "[WALK]" if vmt_data["joy_touch"] else "[IDLE]"
                        vmt_line2 = f"Joy: ({vmt_data['joy_x']:+.2f}, {vmt_data['joy_y']:+.2f}) {walk_state}"
                        cv2.putText(frame, vmt_line2, (bx + 8, by + 164),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 0), 1, cv2.LINE_AA)

                if not seen_left:
                    if vrc_osc.enabled:
                        vrc_osc.send_locomotion(0.0, 0.0)
                    desktop_gestures.update_locomotion(0.0, 0.0)
                if not seen_right:
                    if vrc_osc.enabled:
                        vrc_osc.send_look(0.0)
                    desktop_gestures.update_look(0.0)

                vr_calibrator.prune_stale_hands(active_vr_keys, timestamp=now)

            # Virtual Steering Telemetry when 2 hands are visible (Shared in both modes)
            steering_status = ""
            if len(hands) == 2:
                steer_data = tracker.gesture_recognizer.process_two_hand_steering(hands[0], hands[1])
                angle = steer_data["angle_deg"]
                direction = steer_data["direction"]
                p1 = hands[0]["palm_pixel"]
                p2 = hands[1]["palm_pixel"]
                mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

                if not is_minimized:
                    cv2.line(frame, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.putText(frame, f"STEER: {angle:+.1f} deg ({direction})", (mid_x - 70, mid_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                steering_status = f" | Steer: {angle:+.1f}° {direction}"

            # Display frame only when visible (skipping when minimized prevents DWM stalls)
            if not is_minimized:
                # Top HUD Header Bar
                cv2.rectangle(frame, (0, 0), (w_cam, 36), (15, 15, 25), cv2.FILLED)
                header_text = f"GestureDrive | FPS: {fps:.1f} | Hands: {len(hands)}" + steering_status
                cv2.putText(frame, header_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 230), 2, cv2.LINE_AA)

                # Top Right: Mode Badge & Quick Status
                if mode == AppMode.BASIC:
                    mode_str = "[BASIC: Desktop Mouse]"
                    mode_color = (0, 255, 0)
                    sub_status = f"Scale: {calibrator.motion_scale:.2f}x [+/-] | Mouse: {'Active' if mouse.enabled else 'Paused'}"
                    cv2.putText(frame, mode_str, (w_cam - 325, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)
                    cv2.putText(frame, sub_status, (w_cam - 325, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(frame, "Press [V] for VRChat / VR Mode", (w_cam - 325, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 255), 1, cv2.LINE_AA)
                else:
                    mode_str = f"[ADVANCED: SteamVR ({vmt.mode.name})]"
                    mode_color = (255, 0, 180)
                    mount_name = "Desk" if vr_calibrator.mapper.mount_mode == CameraMountMode.DESK_FACING_USER else "HMD"
                    osc_state = f"ON:{vrc_osc.port}" if vrc_osc.enabled else "OFF"
                    vmt_state = f"ON:{vmt.port}({vmt.packets_sent})" if vmt.enabled else "PAUSED"
                    sub_status = f"VMT {vmt_state} | OSC {osc_state} | Mount: {mount_name} [H]"
                    vrc_fg = desktop_gestures.is_vrchat_foreground()
                    focus_tag = "VRChat: Focused (Shift+Fn Active)" if vrc_fg else "VRChat: Not Focused (Click VRChat)"
                    focus_col = (0, 255, 0) if vrc_fg else (0, 165, 255)
                    cv2.putText(frame, mode_str, (w_cam - 325, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)
                    cv2.putText(frame, sub_status, (w_cam - 325, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(frame, focus_tag, (w_cam - 325, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.34, focus_col, 1, cv2.LINE_AA)

                # Bottom Status Bar & Shortcuts Guide
                cv2.rectangle(frame, (0, h_cam - 28), (w_cam, h_cam), (15, 15, 25), cv2.FILLED)

                if mode == AppMode.BASIC:
                    if calibrator.is_calibrating:
                        pct = int(calibrator.calibration_progress * 100)
                        calib_str = f"CALIBRATING WORKSPACE: {pct}% (Hold hand steady...)"
                        cv2.putText(frame, calib_str, (15, h_cam - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        guide_str = "[V] Mode | [M] Clutch | [C] Calib | [+/-] Scale | [[ / ]] Box | [R] Reset | [Q] Quit"
                        cv2.putText(frame, guide_str, (15, h_cam - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
                        if target_screen_x is not None and target_screen_y is not None:
                            coord_str = f"({target_screen_x}, {target_screen_y})"
                            cv2.putText(frame, coord_str, (w_cam - 110, h_cam - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1, cv2.LINE_AA)
                else:
                    guide_str = "[V] Mode | [T] VMT | [Y] Emul | [O] OSC | [H] Mount | [C] Zero | [R] Reset | [Q] Quit"
                    cv2.putText(frame, guide_str, (15, h_cam - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 200, 255), 1, cv2.LINE_AA)
                    origin = vr_calibrator.origin_offset
                    orig_str = f"Origin: ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}m)"
                    cv2.putText(frame, orig_str, (w_cam - 200, h_cam - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 230), 1, cv2.LINE_AA)

                cv2.imshow(window_name, frame)
                # Standard waitKey(1) when window is actively rendered
                key = cv2.waitKey(1) & 0xFF
            else:
                # Instant non-blocking pollKey (<0.02ms) when window is minimized to prevent 15ms DWM stalls
                key = cv2.pollKey() & 0xFF

            # Keypress handling
            if key == ord('q') or key == 27:
                print("[System] User requested exit.")
                break
            elif key == ord('v') or key == ord('V'):
                if mode == AppMode.BASIC:
                    mode = AppMode.ADVANCED_VR
                    vr_calibrator.pose_tracker.reset()
                    if vmt.enabled:
                        vmt.spawn_devices()
                    print("\n>>> SWITCHED TO: ADVANCED VR MODE (6-DOF Hand Pose & SteamVR VMT / VRChat OSC)")
                else:
                    mode = AppMode.BASIC
                    desktop_gestures.release_all()
                    if vrc_osc.packets_sent > 0:
                        vrc_osc.reset_neutral()
                    vmt.despawn_devices()
                    print("\n>>> SWITCHED TO: BASIC MODE (Desktop PC Mouse Controller)")
            elif key == ord('m') or key == ord('M'):
                if mode == AppMode.BASIC:
                    mouse.toggle_enabled()
            elif key == ord('o') or key == ord('O'):
                vrc_osc.toggle_enabled()
            elif key == ord('t') or key == ord('T'):
                if mode == AppMode.ADVANCED_VR:
                    vmt.toggle_enabled()
            elif key == ord('y') or key == ord('Y'):
                if mode == AppMode.ADVANCED_VR:
                    vmt.toggle_mode()
            elif key == ord('c') or key == ord('C'):
                if mode == AppMode.BASIC:
                    calibrator.start_calibration()
                else:
                    if primary_hand:
                        vr_calibrator.calibrate_origin(primary_hand["landmarks"])
                    else:
                        print("[VR] Please show your hand to calibrate VR origin.")
            elif key in (ord('+'), ord('=')):
                calibrator.adjust_scale(+0.15)
            elif key in (ord('-'), ord('_')):
                calibrator.adjust_scale(-0.15)
            elif key == ord('['):
                calibrator.adjust_margins(-0.02)
            elif key == ord(']'):
                calibrator.adjust_margins(+0.02)
            elif key == ord('h') or key == ord('H'):
                if vr_calibrator.mapper.mount_mode == CameraMountMode.DESK_FACING_USER:
                    vr_calibrator.mapper.mount_mode = CameraMountMode.HMD_MOUNTED
                    print("[VR] Camera Mount Mode set to: HMD MOUNTED (Looking Forward)")
                else:
                    vr_calibrator.mapper.mount_mode = CameraMountMode.DESK_FACING_USER
                    print("[VR] Camera Mount Mode set to: DESK FACING (Looking at User)")
                vr_calibrator.pose_tracker.reset()
            elif key == ord('r') or key == ord('R'):
                cursor_filter.reset()
                precision_curve.reset()
                calibrator.reset_defaults()
                vr_calibrator.reset()
                if mode == AppMode.ADVANCED_VR:
                    vrc_osc.reset_neutral()
                    vmt.reset_neutral()
                hand_lost_time = None
                last_valid_screen_pos = None
                print("[System] Reset all filters, workspace, and VR calibration to defaults.")

    except KeyboardInterrupt:
        print("[System] Terminated by user interrupt.")
    finally:
        print("[System] Cleaning up resources...")
        camera.stop()
        if mouse.is_left_down:
            mouse.left_up()
        if vrc_osc.packets_sent > 0:
            vrc_osc.reset_neutral()
        if vmt.packets_sent > 0 or vmt.enabled:
            vmt.despawn_devices()
        if winmm:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
        desktop_gestures.release_all()
        cv2.destroyAllWindows()
        print("[System] GestureDrive stopped cleanly.")


if __name__ == "__main__":
    main()
