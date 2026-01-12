import numpy as np
import cv2
from collections import deque


class FrameStabilizer:
    def __init__(self, width, height, buffer_size=10):
        self.width = width
        self.height = height
        self.buffer_size = buffer_size

        self.prev_smooth_trajectory = np.zeros(3, dtype=np.float64)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.buffer_size)
        self.cumulative_motion_buffer = deque(maxlen=self.buffer_size)
        self.cumulative_motion = np.zeros(3, dtype=np.float64)

    def stabilize(self, frame):
        self.append_raw_frame(frame)
        if len(self.frame_buffer) > 1: self.estimate_transform()
        smoothed_frame = self.smooth_frame()
        return smoothed_frame

    def smooth_frame(self):
        correction = self.smooth_trajectory() - self.cumulative_motion
        dx_c, dy_c, da_c = correction

        M = self.build_affine(dx_c, dy_c, da_c)

        frame_to_stabilize = self.frame_buffer[-1]
        stabilized = cv2.warpAffine(frame_to_stabilize, M, (self.width, self.height))
        return stabilized

    def append_raw_frame(self, frame):
        self.frame_buffer.append(frame)
        self.grayscale_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def estimate_transform(self):
        prev_pts = cv2.goodFeaturesToTrack(self.grayscale_buffer[-2], 200, 0.01, 30)
        if prev_pts is None:
            return 0, 0, 0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.grayscale_buffer[-2], self.grayscale_buffer[-1], prev_pts, None)
        status = status.reshape(-1)

        prev_pts = prev_pts[status == 1]
        curr_pts = curr_pts[status == 1]

        if len(prev_pts) < 10:
            return 0, 0, 0

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            return 0, 0, 0

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        self.cumulative_motion += np.array([dx, dy, da])
        self.cumulative_motion_buffer.append(self.cumulative_motion.copy())


    def make_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def smooth_trajectory(self, alpha=0.02):
        current = self.cumulative_motion
        smooth = alpha * current + (1 - alpha) * self.prev_smooth_trajectory
        self.prev_smooth_trajectory = smooth
        return smooth

    @staticmethod
    def build_affine(dx, dy, da):
        cos = np.cos(da)
        sin = np.sin(da)
        return np.array([[cos, -sin, dx],
                         [sin,  cos, dy]], dtype=np.float32)
