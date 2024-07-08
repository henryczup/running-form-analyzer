from collections import deque
from typing import Literal, Tuple
import numpy as np
from utils.filters import Filters

class FootStrikeDetector:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "kalman", 
                 window_size: int = 10, detection_axis: Literal["x", "y"] = "x",
                 adaptive_window: int = 50, initial_threshold: float = 0.02):
        self.filter_type = filter_type
        self.window_size = window_size
        self.detection_axis = detection_axis
        self.positions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.last_foot_strike_time = None
        
        self.filter = Filters.create_filter(filter_type, window_size=window_size)

        # Adaptive thresholding parameters
        self.adaptive_window = adaptive_window
        self.motion_history = deque(maxlen=adaptive_window)
        self.threshold = initial_threshold
        self.min_threshold = 0.005  # Minimum threshold to prevent overfitting to noise
        self.max_threshold = 0.1    # Maximum threshold to ensure detection in large movements

    def update(self, ankle_position: np.ndarray, timestamp: float) -> Tuple[bool, float, float]:
        position = ankle_position[0] if self.detection_axis == "x" else ankle_position[1]
        
        if self.filter:
            filtered_position = self.filter.update(position)
        else:
            filtered_position = position
        
        self.positions.append(filtered_position)
        self.timestamps.append(timestamp)
        self.motion_history.append(filtered_position)

        self._update_adaptive_threshold()

        foot_strike_detected = False
        stride_time = 0.0

        if len(self.positions) < 3:
            return foot_strike_detected, stride_time, self.threshold

        # Detect foot strike based on the chosen axis and adaptive threshold
        if self.detection_axis == "x":
            # Foot strike when ankle x-position is at its minimum
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3] and
                self.positions[-3] - self.positions[-2] > self.threshold):
                foot_strike_detected = True
        else:  # "y" axis
            # Foot strike when ankle y-position starts increasing after decreasing
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3] and
                self.positions[-1] - self.positions[-2] > self.threshold):
                foot_strike_detected = True

        if foot_strike_detected:
            if self.last_foot_strike_time is None or (timestamp - self.last_foot_strike_time) > 0.4:  # Prevent false positives
                if self.last_foot_strike_time is not None:
                    stride_time = timestamp - self.last_foot_strike_time
                self.last_foot_strike_time = timestamp
            else:
                foot_strike_detected = False  # Reset if it's too soon after the last detection

        return foot_strike_detected, stride_time, self.threshold

    def _update_adaptive_threshold(self):
        if len(self.motion_history) < self.adaptive_window:
            return

        motion_range = max(self.motion_history) - min(self.motion_history)
        new_threshold = motion_range * 0.1  # 10% of the range as a starting point

        # Apply some smoothing to the threshold update
        self.threshold = 0.7 * self.threshold + 0.3 * new_threshold

        # Ensure the threshold stays within defined bounds
        self.threshold = max(min(self.threshold, self.max_threshold), self.min_threshold)

    def get_filtered_position(self) -> float:
        return self.positions[-1] if self.positions else None