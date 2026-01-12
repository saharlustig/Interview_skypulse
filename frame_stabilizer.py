import numpy as np
import cv2
from collections import deque

MINIMUM_POINTS = 10
MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 30


class FrameStabilizer:
    def __init__(self, width, height, buffer_size=10):
        self.width = width
        self.height = height
        self.buffer_size = buffer_size

        self.prev_smooth_trajectory = np.zeros(3, dtype=np.float64)
        self.cumulative_motion = np.zeros(3, dtype=np.float64)

        self._init_buffers()

    def stabilize(self, frame):
        self.append_raw_frame(frame)
        if len(self.frame_buffer) > 1: self.estimate_frame_transform()
        return self.apply_trajectory_smoothing()

    def estimate_frame_transform(self):
        previous_points = cv2.goodFeaturesToTrack(self.grayscale_buffer[-2], MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE)
        if previous_points is None: return self._append_last_motion_and_return()

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(self.grayscale_buffer[-2], self.grayscale_buffer[-1],
                                                             previous_points, None)
        status = np.squeeze(status)
        previous_points, current_points = (previous_points[status == 1], current_points[status == 1])

        if len(previous_points) < MINIMUM_POINTS: return self._append_last_motion_and_return()

        m, _ = cv2.estimateAffinePartial2D(previous_points, current_points)
        if m is None: return self._append_last_motion_and_return()

        dx = m[0, 2]
        dy = m[1, 2]
        d_theta = np.arctan2(m[1, 0], m[0, 0])

        self.cumulative_motion += np.array([dx, dy, d_theta])
        self.cumulative_motion_buffer.append(self.cumulative_motion.copy())

    def apply_trajectory_smoothing(self):
        if not self.frame_buffer:
            raise ValueError("Frame buffer is empty — cannot stabilize frame.")

        # Compute the difference between smoothed and actual motion
        correction = self.compute_smoothed_motion() - self.cumulative_motion
        dx_c, dy_c, da_c = correction

        M = self.build_affine(dx_c, dy_c, da_c)

        frame_to_stabilize = self.frame_buffer[-1]
        stabilized_frame = cv2.warpAffine(frame_to_stabilize, M, (self.width, self.height))
        return stabilized_frame

    def append_raw_frame(self, frame):
        self.frame_buffer.append(frame)
        self.grayscale_buffer.append(self.make_grayscale(frame))

    def compute_smoothed_motion(self, alpha=0.02):
        current = self.cumulative_motion
        smooth = alpha * current + (1 - alpha) * self.prev_smooth_trajectory
        self.prev_smooth_trajectory = smooth
        return smooth

    def _init_buffers(self):
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.buffer_size)
        self.cumulative_motion_buffer = deque(maxlen=self.buffer_size)

    def _append_last_motion_and_return(self):
        self.cumulative_motion_buffer.append(self.cumulative_motion.copy())

    @staticmethod
    def make_grayscale(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def build_affine(dx, dy, da):
        cos = np.cos(da)
        sin = np.sin(da)
        return np.array([[cos, -sin, dx],
                         [sin, cos, dy]], dtype=np.float32)
