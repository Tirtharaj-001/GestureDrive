"""
vision/hand_tracker.py - Ultra-low-latency MediaPipe Tasks HandLandmarker wrapper for dual hand tracking.
Uses VisionRunningMode.VIDEO with temporal tracking for 60+ FPS real-time responsiveness.
"""

import os
import time
import urllib.request
import cv2
import mediapipe as mp
from typing import List, Dict, Any
from vision.landmarks import LandmarkPoint, WRIST, INDEX_MCP, PINKY_MCP, get_palm_center
from vision.gestures import GestureRecognizer, GestureType


class HandTracker:
    def __init__(self, model_path: str = None, max_hands: int = 2, min_detection_con: float = 0.45, min_track_con: float = 0.45):
        self.max_hands = max_hands
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con
        self.gesture_recognizer = GestureRecognizer()
        self.last_timestamp_ms = 0

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'hand_landmarker.task')

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Downloading Hand Landmarker model to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # VIDEO mode enables temporal landmark tracking between frames, boosting FPS 3-4x over static IMAGE mode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_detection_con,
            min_hand_presence_confidence=self.min_detection_con,
            min_tracking_confidence=self.min_track_con
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Skeleton bone connections for drawing
        self.connections = [
            # Palm
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),
            # Thumb
            (2, 3), (3, 4),
            # Index
            (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20)
        ]

    def process_frame(self, frame_bgr, mirrored: bool = False) -> List[Dict[str, Any]]:
        """
        Runs video tracking inference on frame.
        Set mirrored=True when frame_bgr was horizontally flipped (cv2.flip(frame, 1))
        so MediaPipe's inverted selfie-view Left/Right labels are swapped back to true anatomical hands.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonically increasing millisecond timestamp required for VIDEO mode
        now_ms = int(time.time() * 1000)
        if now_ms <= self.last_timestamp_ms:
            now_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = now_ms

        result = self.landmarker.detect_for_video(mp_image, now_ms)
        hands = []

        if result and result.hand_landmarks:
            seen_labels: Dict[str, int] = {}
            for idx, lm_set in enumerate(result.hand_landmarks):
                landmarks = []
                pixel_coords = []
                for lm in lm_set:
                    landmarks.append(LandmarkPoint(lm.x, lm.y, lm.z))
                    px = max(0, min(w - 1, int(lm.x * w)))
                    py = max(0, min(h - 1, int(lm.y * h)))
                    pixel_coords.append((px, py))

                raw_handedness = "Right"
                confidence = 0.9
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    raw_handedness = result.handedness[idx][0].category_name
                    confidence = float(result.handedness[idx][0].score)

                # Flip Left <-> Right when the input image was mirrored horizontally
                if mirrored and raw_handedness in ("Left", "Right"):
                    handedness = "Right" if raw_handedness == "Left" else "Left"
                else:
                    handedness = raw_handedness

                palm = get_palm_center(landmarks)

                # Debounce gesture per hand identity instead of detection list index
                slot = seen_labels.get(handedness, 0)
                seen_labels[handedness] = slot + 1
                debounce_key = f"{handedness}_{slot}"

                raw_gesture, gest_conf = self.gesture_recognizer.classify_hand_gesture(landmarks, handedness)
                clean_gesture = self.gesture_recognizer.debounce_gesture(debounce_key, raw_gesture)

                hands.append({
                    "id": idx,
                    "landmarks": landmarks,
                    "pixel_coords": pixel_coords,
                    "handedness": handedness,
                    "confidence": confidence,
                    "palm_center": palm,
                    "palm_pixel": (int(palm.x * w), int(palm.y * h)),
                    "gesture": clean_gesture,
                    "gesture_name": clean_gesture.value,
                    "gesture_confidence": gest_conf
                })

        return hands

    def draw_hands(self, frame, hands: List[Dict[str, Any]], color_theme=(0, 255, 230)):
        """Fast cyber skeleton rendering."""
        for hand in hands:
            pixels = hand["pixel_coords"]

            # Draw bones
            for start_idx, end_idx in self.connections:
                pt1 = pixels[start_idx]
                pt2 = pixels[end_idx]
                cv2.line(frame, pt1, pt2, (35, 35, 45), 2, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, color_theme, 1, cv2.LINE_AA)

            # Draw landmark nodes (joints)
            for idx, (px, py) in enumerate(pixels):
                if idx in (4, 8, 12, 16, 20):
                    # Highlight fingertips
                    cv2.circle(frame, (px, py), 5, (255, 0, 128), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 3, color_theme, cv2.FILLED, cv2.LINE_AA)

            # Draw palm center
            ppx, ppy = hand["palm_pixel"]
            cv2.circle(frame, (ppx, ppy), 5, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
