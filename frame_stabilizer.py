import numpy as np
import cv2
from collections import deque


class FrameStabilizer:
    def __init__(self, width, height, buffer_size=10):
        self.width = width
        self.height = height
        self.buffer_size = buffer_size

        self.prev_smooth = np.zeros(3, dtype=np.float64)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.gray_buffer = deque(maxlen=self.buffer_size)
        self.transform_buffer = deque(maxlen=self.buffer_size)
        self.trajectory_buffer = deque(maxlen=self.buffer_size)
        self.cumulative_trajectory = np.zeros(3, dtype=np.float64)

    @property
    def middle_buffer_frame(self):
        return self.buffer_size // 2

    def stabilize(self, frame):
        self.append_raw_frame(frame)
        if len(self.frame_buffer) > 1: self.estimate_transform()
        # if len(self.frame_buffer) == self.buffer_size:
        smoothed_frame = self.smooth_frame()
        return smoothed_frame
        # return frame

    def smooth_frame(self):
        # only output when buffer is full
        # smooth trajectory
        smooth_traj = self.smooth_trajectory()

        # center frame trajectory
        center_traj = self.cumulative_trajectory

        # correction = smoothed - original
        correction = smooth_traj - center_traj
        dx_c, dy_c, da_c = correction

        M = self.build_affine(dx_c, dy_c, da_c)

        frame_to_stabilize = self.frame_buffer[-1]
        stabilized = cv2.warpAffine(frame_to_stabilize, M, (self.width, self.height))
        return stabilized

    def append_raw_frame(self, frame):
        self.frame_buffer.append(frame)
        self.gray_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def estimate_transform(self):
        prev_pts = cv2.goodFeaturesToTrack(self.gray_buffer[-2], 200, 0.01, 30)
        if prev_pts is None:
            return 0, 0, 0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.gray_buffer[-2], self.gray_buffer[-1], prev_pts, None)
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

        self.cumulative_trajectory += np.array([dx, dy, da])
        self.trajectory_buffer.append(self.cumulative_trajectory.copy())


    def make_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def smooth_trajectory(self, alpha=0.02):
        current = self.cumulative_trajectory
        smooth = alpha * current + (1 - alpha) * self.prev_smooth
        self.prev_smooth = smooth
        return smooth

    @staticmethod
    def build_affine(dx, dy, da):
        cos = np.cos(da)
        sin = np.sin(da)
        return np.array([[cos, -sin, dx],
                         [sin,  cos, dy]], dtype=np.float32)
