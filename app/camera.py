"""
app/camera.py - Threaded low-latency webcam capture stream.
Decouples frame acquisition from inference to prevent buffer bloat and minimize input latency.
"""

import cv2
import threading
import time
from typing import Optional, Tuple

class CameraStream:
    def __init__(self, target_indices=(0, 1, 2, 3), width: int = 640, height: int = 480):
        self.target_indices = target_indices
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_index = -1
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        
        self._init_camera()

    def _init_camera(self):
        for idx in self.target_indices:
            temp = cv2.VideoCapture(idx)
            if temp.isOpened():
                temp.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                temp.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                temp.set(cv2.CAP_PROP_FPS, 60)
                success, frame = temp.read()
                if success and frame is not None:
                    self.cap = temp
                    self.active_index = idx
                    self.frame = frame
                    self.grabbed = True
                    print(f"[Camera] Successfully initialized webcam at index {idx} ({self.width}x{self.height})")
                    break
                else:
                    temp.release()
            else:
                temp.release()
                
        if self.cap is None:
            print("[Camera] WARNING: No physical webcam detected. Simulated mode or fallback available.")

    def start(self):
        if self.cap is None:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = grabbed
            time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[any]]:
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
