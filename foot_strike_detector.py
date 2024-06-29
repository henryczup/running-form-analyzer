from collections import deque
from typing import Literal, Tuple

import numpy as np

from filter import KalmanFilter

class FootStrikeDetector:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "kalman", 
                 window_size: int = 10, smoothing_window: int = 5,
                 detection_axis: Literal["x", "y"] = "x"):
        self.filter_type = filter_type
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.detection_axis = detection_axis
        self.positions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.last_foot_strike_time = None
        
        if self.filter_type == "kalman":
            self.kalman = KalmanFilter(
                initial_state=0,
                initial_estimate_error=1,
                measurement_noise=0.1,
                process_noise=0.01
            )

    def temporal_filter(self, new_position: float) -> float:
        """Apply a simple moving average filter to the new position."""
        self.positions.append(new_position)
        if len(self.positions) < self.smoothing_window:
            return new_position
        return np.mean(list(self.positions)[-self.smoothing_window:])

    def update(self, ankle_position: np.ndarray, timestamp: float) -> Tuple[bool, float]:
        position = ankle_position[0] if self.detection_axis == "x" else ankle_position[1]
        
        if self.filter_type == "temporal":
            filtered_position = self.temporal_filter(position)
        elif self.filter_type == "kalman":
            filtered_position = self.kalman.update(position)
        else:  # no filter
            filtered_position = position
        
        self.positions.append(filtered_position)
        self.timestamps.append(timestamp)

        foot_strike_detected = False
        stride_time = 0.0

        if len(self.positions) < 3:
            return foot_strike_detected, stride_time

        # Detect foot strike based on the chosen axis
        if self.detection_axis == "x":
            # Foot strike when ankle x-position is at its minimum
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3]):
                foot_strike_detected = True
        else:  # "y" axis
            # Foot strike when ankle y-position starts increasing after decreasing
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3]):
                foot_strike_detected = True

        if foot_strike_detected:
            if self.last_foot_strike_time is None or (timestamp - self.last_foot_strike_time) > 0.4:  # Prevent false positives
                if self.last_foot_strike_time is not None:
                    stride_time = timestamp - self.last_foot_strike_time
                self.last_foot_strike_time = timestamp
            else:
                foot_strike_detected = False  # Reset if it's too soon after the last detection

        return foot_strike_detected, stride_time

    def get_filtered_position(self) -> float:
        return self.positions[-1] if self.positions else None

    def set_filter_type(self, filter_type: Literal["temporal", "kalman", "none"]):
        if filter_type not in ["temporal", "kalman", "none"]:
            raise ValueError("Filter type must be either 'temporal', 'kalman', or 'none'")
        self.filter_type = filter_type
        if filter_type == "kalman":
            self.kalman = KalmanFilter(
                initial_state=self.get_filtered_position() or 0,
                initial_estimate_error=1,
                measurement_noise=0.1,
                process_noise=0.01
            )

    def set_detection_axis(self, detection_axis: Literal["x", "y"]):
        if detection_axis not in ["x", "y"]:
            raise ValueError("Detection axis must be either 'x' or 'y'")
        self.detection_axis = detection_axis