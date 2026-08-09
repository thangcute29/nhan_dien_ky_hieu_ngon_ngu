# mobile_app/src/virtual_cam/virtual_camera.py
import cv2
import numpy as np

class VirtualCamera:
    def __init__(self, width=640, height=480, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        # Sử dụng thư viện pyvirtualcam nếu có
        try:
            import pyvirtualcam
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
            self.use_virtual = True
        except Exception as e:
            print(f"Virtual camera not available ({e}), falling back to OpenCV window.")
            self.use_virtual = False

    def send_frame(self, frame):
        if self.use_virtual:
            frame_resized = cv2.resize(frame, (self.width, self.height))
            self.cam.send(frame_resized)

    def close(self):
        if self.use_virtual:
            self.cam.close()