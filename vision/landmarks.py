"""
vision/landmarks.py - Mathematical calculations, angles, finger extension, and spatial heuristics for hand landmarks.
"""

import math
from typing import List, Tuple, Dict

# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

class LandmarkPoint:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

def dist_2d(p1, p2) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def dist_3d(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle_deg(a, b, c) -> float:
    """Calculates angle in degrees at vertex b formed by points a, b, c."""
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])
    
    if mag_ab * mag_cb == 0:
        return 0.0
        
    cos_val = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_val))

def get_palm_center(landmarks) -> LandmarkPoint:
    """Computes palm center from Wrist, Index MCP, and Pinky MCP."""
    w = landmarks[WRIST]
    i = landmarks[INDEX_MCP]
    p = landmarks[PINKY_MCP]
    return LandmarkPoint((w.x + i.x + p.x) / 3.0, (w.y + i.y + p.y) / 3.0, (w.z + i.z + p.z) / 3.0)

def get_hand_scale(landmarks) -> float:
    """Estimates the hand bounding scale (distance from wrist to middle MCP)."""
    return dist_2d(landmarks[WRIST], landmarks[MIDDLE_MCP])

def get_finger_states(landmarks, handedness: str = "Right") -> Dict[str, bool]:
    """
    Determines whether each of the 5 fingers is extended.
    Uses distance from wrist and knuckle joint positions.
    """
    wrist = landmarks[WRIST]
    scale = max(0.01, get_hand_scale(landmarks))
    
    states = {}
    
    # Non-thumb fingers: extended if TIP is farther from wrist than PIP and DIP
    finger_names = ["index", "middle", "ring", "pinky"]
    tip_indices = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pip_indices = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    mcp_indices = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    
    for name, tip_idx, pip_idx, mcp_idx in zip(finger_names, tip_indices, pip_indices, mcp_indices):
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        
        d_tip = dist_2d(tip, wrist)
        d_pip = dist_2d(pip, wrist)
        d_mcp = dist_2d(mcp, wrist)
        
        # Extended if tip is clearly beyond the pip joint
        is_extended = (d_tip > d_pip) and (d_pip > d_mcp * 0.9)
        states[name] = bool(is_extended)
        
    # Thumb: Extended when opened away from index MCP and farther from wrist than IP joint
    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    
    d_tip_index = dist_2d(thumb_tip, index_mcp)
    d_mcp_index = dist_2d(thumb_mcp, index_mcp)
    d_tip_wrist = dist_2d(thumb_tip, wrist)
    d_ip_wrist = dist_2d(thumb_ip, wrist)
    
    states["thumb"] = bool(d_tip_index > d_mcp_index * 1.15 and d_tip_wrist > d_ip_wrist)
    
    return states

def calculate_two_hand_steering_angle(hand1_center: LandmarkPoint, hand2_center: LandmarkPoint) -> Tuple[float, LandmarkPoint, LandmarkPoint]:
    """
    Calculates virtual steering wheel rotation angle (-90 to +90 degrees)
    between two hands. Returns (angle_deg, left_hand, right_hand).
    """
    if hand1_center.x < hand2_center.x:
        left_hand = hand1_center
        right_hand = hand2_center
    else:
        left_hand = hand2_center
        right_hand = hand1_center
        
    dx = right_hand.x - left_hand.x
    dy = right_hand.y - left_hand.y
    
    angle_rad = math.atan2(dy, max(0.001, dx))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, left_hand, right_hand
