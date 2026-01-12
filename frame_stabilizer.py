import cv2
import numpy as np
from collections import deque

MAX_CORNERS = 200  # Maximum number of corners to detect in each frame using goodFeaturesToTrack
MINIMUM_POINTS = 10  # Minimum number of valid tracked points required to estimate a reliable transform
MIN_DISTANCE = 30  # Minimum Euclidean distance (in pixels) between detected corners to avoid clustering
QUALITY_LEVEL = 0.01  # Minimum accepted quality of image corners (relative to the best corner); lower = more corners


class FrameStabilizer:
    def __init__(self, width: int, height: int, buffer_size=10) -> None:
        """
        Parameters
        ----------
        width : int
            Width of the video frames to be stabilized.
        height : int
            Height of the video frames to be stabilized.
        buffer_size : int, optional
            Maximum number of frames/motions stored in buffers.
        """
        self.width = width
        self.height = height
        self.buffer_size = buffer_size

        self.previous_smooth_trajectory = np.zeros(3, dtype=np.float64)
        self.cumulative_motion = np.zeros(3, dtype=np.float64)

        self._init_buffers()

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilizes a single video frame using motion estimation and trajectory smoothing.

        Parameters
        ----------
        frame: np.ndarray of shape (height, width, 3)
            The current video frame (in BGR format) to be stabilized.

        Returns
        -------
        np.ndarray of shape (height, width, 3)
            The stabilized version of the input frame, with camera motion compensated.
        """
        self.append_raw_frame(frame)
        if len(self.frame_buffer) > 1: self.estimate_frame_transform()
        return self.apply_trajectory_smoothing()

    def estimate_frame_transform(self) -> None:
        """
        Estimates the affine transform between the two most recent grayscale frames
        using tracked feature points, and updates the cumulative motion buffer.

        This method:
        - Detects good features to track in the previous frame.
        - Tracks those features to the current frame using optical flow.
        - Filters out unreliable points based on tracking status.
        - Estimates a partial affine transform (translation (dx, dy) and rotation d_alpha).
        - Updates the cumulative motion vector and appends it to the motion buffer.

        If feature detection, tracking, or transform estimation fails, the last known
        cumulative motion is reused to maintain buffer consistency.

        Returns
        -------
        None
            This method updates internal state but does not return a value.
        """
        # Detect good features to track in the previous grayscale frame
        previous_points = cv2.goodFeaturesToTrack(self.grayscale_buffer[-2], MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE)
        if previous_points is None: return self._append_last_motion_and_return()

        # Track features to the current frame using optical flow
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(self.grayscale_buffer[-2], self.grayscale_buffer[-1],
                                                             previous_points, None)
        status = np.squeeze(status)  # Flatten the array

        # Filter out points that failed to track
        previous_points, current_points = (previous_points[status == 1], current_points[status == 1])

        if len(previous_points) < MINIMUM_POINTS: return self._append_last_motion_and_return()

        # Estimate affine transform (translation + rotation only)
        transformation_matrix, _ = cv2.estimateAffinePartial2D(previous_points, current_points)
        if transformation_matrix is None: return self._append_last_motion_and_return()

        dx = transformation_matrix[0, 2]
        dy = transformation_matrix[1, 2]
        d_theta = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])

        self.cumulative_motion += np.array([dx, dy, d_theta])
        self.cumulative_motion_buffer.append(self.cumulative_motion.copy())

    def compute_smoothed_motion(self, alpha: float = 0.02) -> np.ndarray:
        """
        Computes an exponentially smoothed version of the current cumulative motion vector.

        This method blends the current cumulative motion with the previously smoothed trajectory
        using exponential smoothing. It helps reduce jitter and noise in the estimated motion
        by favoring historical stability over sudden changes.

        Parameters
        ----------
        alpha : float, optional
            Smoothing factor between 0 and 1. A lower value results in smoother motion but slower
            responsiveness to changes. Default is 0.02.

        Returns
        -------
        np.ndarray of shape (3,)
            The updated smoothed motion vector [dx, dy, d_theta], where dx and dy are translations and
            da is the rotation angle in radians.
        """
        current = self.cumulative_motion
        smooth = alpha * current + (1 - alpha) * self.previous_smooth_trajectory
        self.previous_smooth_trajectory = smooth
        return smooth

    def apply_trajectory_smoothing(self) -> np.ndarray:

        """
        Applies trajectory smoothing to the most recent frame using motion correction.

        This method computes the difference between the smoothed motion trajectory and the
        actual cumulative motion, constructs a corrective affine transform, and applies it
        to the latest frame in the buffer to produce a stabilized output.

        Returns
        -------
        np.ndarray of shape (height, width, 3)
            The stabilized video frame with camera motion compensated.
        """
        if not self.frame_buffer:
            raise ValueError("Frame buffer is empty — cannot stabilize frame.")

        # Compute the difference between smoothed and actual motion
        correction = self.compute_smoothed_motion() - self.cumulative_motion
        dx_c, dy_c, da_c = correction

        transformation_matrix = self.build_affine_transform(dx_c, dy_c, da_c)

        frame_to_stabilize = self.frame_buffer[-1]
        stabilized_frame = cv2.warpAffine(frame_to_stabilize, transformation_matrix, (self.width, self.height))
        return stabilized_frame

    def append_raw_frame(self, frame: np.ndarray) -> None:
        self.frame_buffer.append(frame)
        self.grayscale_buffer.append(self.make_grayscale(frame))

    def _append_last_motion_and_return(self) -> None:
        self.cumulative_motion_buffer.append(self.cumulative_motion.copy())

    def _init_buffers(self) -> None:
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.grayscale_buffer = deque(maxlen=self.buffer_size)
        self.cumulative_motion_buffer = deque(maxlen=self.buffer_size)

    @staticmethod
    def make_grayscale(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def build_affine_transform(dx: float, dy: float, d_theta: float) -> np.ndarray:
        """
        Constructs a 2D affine transformation matrix for translation and rotation.

        This matrix can be used to apply a rigid-body transformation (rotation + translation)
        to an image or a set of points in 2D space.

        Parameters
        ----------
        dx : float
            Translation along the x-axis.
        dy : float
            Translation along the y-axis.
        d_theta : float
            Rotation angle in radians (counterclockwise).

        Returns
        -------
        np.ndarray of shape (2, 3)
            The affine transformation matrix:
            [[cos(theta), -sin(theta), dx],
             [sin(theta),  cos(theta), dy]]
        """
        cos = np.cos(d_theta)
        sin = np.sin(d_theta)
        return np.array([[cos, -sin, dx],
                         [sin, cos, dy]], dtype=np.float32)
