"""
vision/hand_tracker.py - High-speed MediaPipe Tasks HandLandmarker wrapper for dual hand tracking.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from typing import List, Dict, Any
from vision.landmarks import LandmarkPoint, WRIST, INDEX_MCP, PINKY_MCP, get_palm_center
from vision.gestures import GestureRecognizer, GestureType

class HandTracker:
    def __init__(self, model_path: str = None, max_hands: int = 2, min_detection_con: float = 0.6, min_track_con: float = 0.6):
        self.max_hands = max_hands
        self.min_detection_con = min_detection_con
        self.min_track_con = min_track_con
        self.gesture_recognizer = GestureRecognizer()
        
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
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
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

    def process_frame(self, frame_bgr) -> List[Dict[str, Any]]:
        """
        Runs inference on an OpenCV BGR frame and returns structured hand information.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.landmarker.detect(mp_image)
        hands = []
        
        if result and result.hand_landmarks:
            for idx, lm_set in enumerate(result.hand_landmarks):
                landmarks = []
                pixel_coords = []
                for lm in lm_set:
                    landmarks.append(LandmarkPoint(lm.x, lm.y, lm.z))
                    px = max(0, min(w - 1, int(lm.x * w)))
                    py = max(0, min(h - 1, int(lm.y * h)))
                    pixel_coords.append((px, py))
                    
                handedness = "Right"
                confidence = 0.9
                if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                    handedness = result.handedness[idx][0].category_name
                    confidence = float(result.handedness[idx][0].score)
                    
                palm = get_palm_center(landmarks)
                
                # Classify hand gesture & debounce across frames
                raw_gesture, gest_conf = self.gesture_recognizer.classify_hand_gesture(landmarks, handedness)
                clean_gesture = self.gesture_recognizer.debounce_gesture(idx, raw_gesture)
                
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
        """Renders futuristic cyber bones and joints on frame."""
        for hand in hands:
            pixels = hand["pixel_coords"]
            
            # Draw connections (bones)
            for start_idx, end_idx in self.connections:
                pt1 = pixels[start_idx]
                pt2 = pixels[end_idx]
                cv2.line(frame, pt1, pt2, (40, 40, 50), 3, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, color_theme, 1, cv2.LINE_AA)
                
            # Draw landmark nodes (joints)
            for idx, (px, py) in enumerate(pixels):
                # Highlight fingertips
                if idx in [4, 8, 12, 16, 20]:
                    cv2.circle(frame, (px, py), 6, (255, 0, 128), cv2.FILLED, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (px, py), 4, color_theme, cv2.FILLED, cv2.LINE_AA)
                    
            # Draw palm center
            ppx, ppy = hand["palm_pixel"]
            cv2.circle(frame, (ppx, ppy), 5, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
