"""
app/main.py - Day 3 Webcam & Hand Tracking Core Application.
Initializes real-time camera feed, MediaPipe 21 3D hand landmark detection,
wrist and palm center tracking, hand velocity, finger states, and HUD overlay.
"""

import sys
import os
import time
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
from vision.landmarks import get_finger_states, get_hand_scale, get_palm_center

def main():
    print("==========================================================")
    print("  GESTUREDRIVE —  Webcam & Hand Tracking Core Engine")
    print("==========================================================")
    print("  Press 'Q' or 'ESC' in the video window to quit.")
    print("==========================================================\n")

    # Initialize Threaded Camera Stream & MediaPipe Tracker
    camera = CameraStream(width=640, height=480).start()
    tracker = HandTracker(max_hands=2, min_detection_con=0.6, min_track_con=0.6)

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    prev_positions = {}

    try:
        while True:
            grabbed, frame = camera.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            # Mirror frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            h_cam, w_cam, _ = frame.shape

            # Calculate real-time FPS
            frame_count += 1
            now = time.time()
            dt = now - fps_timer
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_timer = now

            # Process hand landmarks using MediaPipe Tasks API
            hands = tracker.process_frame(frame)

            # Draw futuristic cyber skeleton connections & landmark nodes
            tracker.draw_hands(frame, hands)

            # Process detected hand positions & render HUD telemetry
            gesture_summary = []
            for hand in hands:
                hand_id = hand["id"]
                handedness = hand["handedness"]
                conf = hand["confidence"]
                px, py = hand["palm_pixel"]
                landmarks = hand["landmarks"]
                gesture_name = hand.get("gesture_name", "None")
                gesture_summary.append(f"{handedness}: {gesture_name}")

                # Finger status analysis (extended vs curled)
                f_states = get_finger_states(landmarks, handedness)
                ext_count = sum(1 for v in f_states.values() if v)

                # Hand scale calculation
                scale = get_hand_scale(landmarks)

                # Velocity estimate
                velocity = (0.0, 0.0)
                if hand_id in prev_positions and dt > 0:
                    prev_px, prev_py = prev_positions[hand_id]
                    velocity = ((px - prev_px) / dt, (py - prev_py) / dt)
                prev_positions[hand_id] = (px, py)

                # Telemetry HUD Badge
                badge_w, badge_h = 175, 82
                bx = max(10, min(w_cam - badge_w - 10, px - 85))
                by = max(50, min(h_cam - badge_h - 10, py - 95))

                # Dynamic badge border color based on gesture
                border_color = (0, 255, 200) # Default cyan
                if gesture_name == "Open Palm":
                    border_color = (0, 255, 0) # Green for Go
                elif gesture_name == "Fist":
                    border_color = (0, 0, 255) # Red for Brake
                elif gesture_name == "Pinch":
                    border_color = (255, 0, 255) # Magenta for Action
                elif gesture_name == "Pointing":
                    border_color = (255, 255, 0) # Yellow for Point

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

            # Virtual Steering Telemetry when 2 hands are visible
            steering_status = ""
            if len(hands) == 2:
                steer_data = tracker.gesture_recognizer.process_two_hand_steering(hands[0], hands[1])
                angle = steer_data["angle_deg"]
                direction = steer_data["direction"]
                p1 = hands[0]["palm_pixel"]
                p2 = hands[1]["palm_pixel"]
                mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

                # Render virtual steering axle
                cv2.line(frame, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 255), cv2.FILLED, cv2.LINE_AA)
                cv2.putText(frame, f"STEER: {angle:+.1f} deg ({direction})", (mid_x - 70, mid_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                steering_status = f" | Steer: {angle:+.1f}° {direction}"

            # Top HUD Header Bar
            cv2.rectangle(frame, (0, 0), (w_cam, 36), (15, 15, 25), cv2.FILLED)
            header_text = f"GestureDrive | FPS: {fps:.1f} | Hands: {len(hands)}" + steering_status
            cv2.putText(frame, header_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 230), 2, cv2.LINE_AA)

            # Display OpenCV window
            cv2.imshow("GestureDrive — Day 3 Hand Tracking Core", frame)

            # Keypress handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                print("[System] User requested exit.")
                break

    except KeyboardInterrupt:
        print("[System] Terminated by user interrupt.")
    finally:
        print("[System] Cleaning up camera and window resources...")
        camera.stop()
        cv2.destroyAllWindows()
        print("[System] GestureDrive Day 3 engine stopped cleanly.")

if __name__ == "__main__":
    main()
