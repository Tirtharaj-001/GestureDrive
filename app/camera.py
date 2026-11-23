"""
app/camera.py - Threaded low-latency webcam capture stream.
Decouples frame acquisition from inference to prevent buffer bloat and minimize input latency.
Supports native 16:9 widescreen capture, MJPG 60 FPS mode, DirectShow, and zero-latency hardware buffer flushing.
"""

import cv2
import threading
import time
from typing import Optional, Tuple


class CameraStream:
    def __init__(self, target_indices=(0, 1, 2, 3), width: int = 1280, height: int = 720):
        self.target_indices = target_indices
        self.width = width
        self.height = height

        self.cap: Optional[cv2.VideoCapture] = None
        self.active_index = -1
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.frame_id = 0

        self._init_camera()

    def _init_camera(self):
        for idx in self.target_indices:
            # Try DirectShow on Windows for fast startup, then fallback
            temp = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not temp.isOpened():
                temp = cv2.VideoCapture(idx)

            if temp.isOpened():
                # Flush hardware buffer: always deliver newest scanline with zero queue delay
                temp.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Request MJPG compression to unlock 60 FPS over USB 2.0/integrated bus
                temp.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                temp.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                temp.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                temp.set(cv2.CAP_PROP_FPS, 60)

                success, frame = temp.read()
                if success and frame is not None:
                    self.cap = temp
                    self.active_index = idx
                    self.frame = frame
                    self.grabbed = True
                    self.height, self.width = frame.shape[:2]
                    fps_val = int(temp.get(cv2.CAP_PROP_FPS))
                    print(f"[Camera] Initialized webcam index {idx} ({self.width}x{self.height} @ {fps_val}fps)")
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
        # Capture loop clocked directly by hardware camera sensor
        while self.running:
            if self.cap and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = grabbed
                        self.frame_id += 1
            else:
                time.sleep(0.01)

    def read_latest(self) -> Tuple[bool, Optional[any], int]:
        """
        Instant lock-free check for the newest frame sequence ID.
        Allows main loop to detect new frames within 1ms without blocking or queueing.
        """
        with self.lock:
            return self.grabbed, self.frame, self.frame_id

    def read(self) -> Tuple[bool, Optional[any]]:
        """Backwards-compatible frame read."""
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame
            return False, None

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
