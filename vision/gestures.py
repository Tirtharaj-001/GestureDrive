"""
vision/gestures.py - Single-hand & dual-hand gesture recognition engine.
Classifies hand poses using MediaPipe 3D landmarks, finger extension heuristics, and distance vectors.
Includes temporal debouncing for smooth gesture transitions.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
from vision.landmarks import (
    LandmarkPoint,
    WRIST, THUMB_TIP, THUMB_IP, THUMB_MCP,
    INDEX_TIP, INDEX_PIP, INDEX_MCP,
    MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP,
    RING_TIP, RING_PIP, RING_MCP,
    PINKY_TIP, PINKY_PIP, PINKY_MCP,
    dist_2d, get_finger_states, get_hand_scale,
    calculate_two_hand_steering_angle
)

class GestureType(Enum):
    NONE = "None"
    FIST = "Fist"                # All fingers curled -> Brake / Stop
    OPEN_PALM = "Open Palm"      # All fingers extended -> Accelerate / Go
    PINCH = "Pinch"              # Thumb & Index tips touching -> Select / Action
    OK = "OK"                    # Thumb & Index touching with other fingers extended -> Confirm
    POINTING = "Pointing"        # Only Index finger extended -> Point / Aim
    PEACE = "Peace"              # Index & Middle extended -> Boost / Special
    THUMBS_UP = "Thumbs Up"      # Thumb extended upwards -> Confirm / Horn
    THUMBS_DOWN = "Thumbs Down"  # Thumb extended downwards -> Cancel / Reverse
    THREE_FINGERS = "Three Fingers" # Index, Middle, Ring extended -> Option 3
    ROCK = "Rock On"             # Index & Pinky extended -> Action / Horn

class GestureRecognizer:
    def __init__(self, pinch_threshold_ratio: float = 0.35, history_len: int = 2):
        """
        :param pinch_threshold_ratio: Ratio of hand scale to register pinch gesture.
        :param history_len: Frames history window for temporal gesture debouncing.
        """
        self.pinch_threshold_ratio = pinch_threshold_ratio
        self.history_len = history_len
        self._gesture_history: Dict[Any, List[GestureType]] = {}

    def classify_hand_gesture(self, landmarks: List[LandmarkPoint], handedness: str = "Right") -> Tuple[GestureType, float]:
        """
        Classifies single hand posture based on landmarks and spatial heuristics.
        Returns (GestureType, confidence_score).
        """
        if not landmarks or len(landmarks) < 21:
            return GestureType.NONE, 0.0

        f_states = get_finger_states(landmarks, handedness)
        thumb = f_states.get("thumb", False)
        index = f_states.get("index", False)
        middle = f_states.get("middle", False)
        ring = f_states.get("ring", False)
        pinky = f_states.get("pinky", False)

        scale = max(0.01, get_hand_scale(landmarks))

        # Check Pinch / OK Gesture (distance between THUMB_TIP and INDEX_TIP)
        d_thumb_index = dist_2d(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        if d_thumb_index < scale * self.pinch_threshold_ratio:
            # If remaining fingers (middle, ring, pinky) are extended -> OK sign
            if middle and (ring or pinky):
                return GestureType.OK, 0.95
            return GestureType.PINCH, 0.95

        # Check Fist (all 4 non-thumb fingers folded, thumb folded or neutral)
        if not index and not middle and not ring and not pinky:
            # Check for Thumbs Up / Thumbs Down if thumb is extended
            wrist = landmarks[WRIST]
            thumb_tip = landmarks[THUMB_TIP]
            thumb_mcp = landmarks[THUMB_MCP]
            
            if thumb and (thumb_tip.y < thumb_mcp.y) and (thumb_tip.y < wrist.y):
                return GestureType.THUMBS_UP, 0.90
            elif thumb and (thumb_tip.y > thumb_mcp.y) and (thumb_tip.y > wrist.y):
                return GestureType.THUMBS_DOWN, 0.90
            else:
                return GestureType.FIST, 0.95

        # Check Open Palm (all 5 fingers extended, or at least 4 non-thumb extended)
        if index and middle and ring and pinky:
            return GestureType.OPEN_PALM, 0.95

        # Check Pointing (only index extended)
        if index and not middle and not ring and not pinky:
            return GestureType.POINTING, 0.90

        # Check Peace / Victory (index and middle extended, ring and pinky folded)
        if index and middle and not ring and not pinky:
            return GestureType.PEACE, 0.90

        # Check Rock On (index and pinky extended, middle and ring folded)
        if index and pinky and not middle and not ring:
            return GestureType.ROCK, 0.88

        # Check Three Fingers (index, middle, ring extended)
        if index and middle and ring and not pinky:
            return GestureType.THREE_FINGERS, 0.85

        return GestureType.NONE, 0.50

    def debounce_gesture(self, hand_id: Any, raw_gesture: GestureType) -> GestureType:
        """
        Applies a temporal sliding window filter to stabilize gesture outputs.
        """
        if hand_id not in self._gesture_history:
            self._gesture_history[hand_id] = []

        history = self._gesture_history[hand_id]
        history.append(raw_gesture)

        if len(history) > self.history_len:
            history.pop(0)

        # Count occurrences in history window
        counts: Dict[GestureType, int] = {}
        for g in history:
            counts[g] = counts.get(g, 0) + 1

        # Return majority vote gesture
        majority_gesture = max(counts, key=counts.get)
        return majority_gesture

    def process_two_hand_steering(self, hand1: Dict[str, Any], hand2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes dual hand telemetry to estimate virtual steering wheel angle & direction.
        """
        c1 = hand1["palm_center"]
        c2 = hand2["palm_center"]

        angle_deg, left_hand, right_hand = calculate_two_hand_steering_angle(c1, c2)

        # Categorize direction
        if angle_deg < -10.0:
            direction = "LEFT"
        elif angle_deg > 10.0:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        # Steering intensity (0.0 to 1.0 normalized based on max 45 degree turn)
        intensity = min(1.0, abs(angle_deg) / 45.0)

        return {
            "angle_deg": angle_deg,
            "direction": direction,
            "intensity": intensity,
            "left_palm": left_hand,
            "right_palm": right_hand
        }
