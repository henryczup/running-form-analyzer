
from typing import Any, Dict, List, Literal, Tuple
import numpy as np

from io_utils.display import display_dev_mode, display_user_mode
from metrics.recommendations import Recommendation
from metrics.angle import Angle
from metrics.step import StepMetrics
from metrics.distance import Distance
from utils.utils import get_valid_keypoints

class Calculator:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "temporal",
                 detection_axis: Literal["x", "y"] = "y"):
        self.angle_metrics = Angle()
        self.step_metrics = StepMetrics(filter_type, detection_axis)
        self.distance_metrics = Distance()
        self.recommendations_calculator = Recommendation(window_size=30, consistency_threshold=0.7)
        self.start_time = None

        self.available_metrics = {
            'trunk_angle': 0.0,
            'trunk_angle_assessment': '',
            'left_knee_angle': 0.0,
            'right_knee_angle': 0.0,
            'left_knee_assessment': '',
            'right_knee_assessment': '',
            'max_left_arm_swing_angle': 0.0,
            'max_left_arm_swing_angle_assessment': '',
            'left_elbow_angle': 0.0,
            'left_elbow_angle_assessment': '',
            'distance_cm': 0.0,
            'vertical_oscillation': 0.0,
            'vertical_oscillation_assessment': '',
            'left_hip_ankle_angle': 0.0,
            'right_hip_ankle_angle': 0.0,
            'left_hip_ankle_angle_assessment': '',
            'right_hip_ankle_angle_assessment': '',
            'left_foot_strike': False,
            'right_foot_strike': False,
            'left_ankle_position': 0.0,
            'right_ankle_position': 0.0,
            'total_step_count': 0,
            'steps_per_minute': 0.0,
            'elapsed_time': 0.0,
            'recommendations': [],
        }

    def calculate_metrics(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], frame: np.ndarray, 
                          confidence_threshold: float, timestamp: float, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.start_time is None:
            self.start_time = timestamp

        metrics = {metric: default_value for metric, default_value in self.available_metrics.items()}
        metrics['elapsed_time'] = timestamp - self.start_time

        valid_keypoints = get_valid_keypoints(keypoint_coords, keypoint_confs, confidence_threshold)

        self.angle_metrics.calculate(valid_keypoints, metrics)
        self.step_metrics.calculate(valid_keypoints, timestamp, metrics)
        self.distance_metrics.calculate(valid_keypoints, metrics)

        metrics['recommendations'] = self.recommendations_calculator.get_recommendations(metrics)

        if mode == "dev":
            display_dev_mode(frame, metrics)
        elif mode == "user":
            display_user_mode(frame, metrics)

        return frame, metrics

    def get_available_metrics(self) -> Dict[str, Any]:
        return self.available_metrics