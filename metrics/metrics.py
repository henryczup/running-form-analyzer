
from typing import Any, Dict, List, Literal, Tuple
import numpy as np
from core.config import Config
from feedback.audio_feedback import AudioFeedbackProvider
from metrics.angle_metrics import AngleMetrics
from feedback.recommendations import Recommendation
from metrics.distance_metrics import DistanceMetrics

class Metrics:
    def __init__(self, config: Config):
        self.config = config
        self.angle_metrics = AngleMetrics(config)
        self.distance_metrics = DistanceMetrics(config)
        self.audio_provider = AudioFeedbackProvider()
        self.recommendations_calculator = Recommendation(window_size=30, consistency_threshold=0.7, audio_provider=self.audio_provider)
        self.start_time = None

        self.key_metrics = {
            'head_angle': 0.0,
            'head_angle_assessment': '',

            'trunk_angle': 0.0,
            'trunk_angle_assessment': '',

            'left_elbow_angle': 0.0,
            'right_elbow_angle': 0.0,
            'left_elbow_angle_assessment': '',
            'right_elbow_angle_assessment': '',

            'left_hip_ankle_angle_at_strike': 0.0,
            'right_hip_ankle_angle_at_strike': 0.0,
            'left_hip_ankle_angle_assessment': '',
            'right_hip_ankle_angle_assessment': '',

            'left_knee_angle_at_strike': 0.0,
            'right_knee_angle_at_strike': 0.0,
            'left_knee_assessment': '',
            'right_knee_assessment': '',

            'left_shank_angle_at_strike': 0.0,
            'right_shank_angle_at_strike': 0.0,
            'left_shank_assessment': '',
            'right_shank_assessment': '',

            'max_left_arm_backward_swing': 0.0,
            'max_left_arm_forward_swing': 0.0,
            'max_right_arm_backward_swing': 0.0,
            'max_right_arm_forward_swing': 0.0,
            'left_arm_backward_swing_assessment': '',
            'left_arm_forward_swing_assessment': '',
            'right_arm_backward_swing_assessment': '',
            'right_arm_forward_swing_assessment': '',

            'max_left_hip_backward_swing': 0.0,
            'max_left_hip_forward_swing': 0.0,
            'max_right_hip_backward_swing': 0.0,
            'max_right_hip_forward_swing': 0.0,
            'left_hip_backward_swing_assessment': '',
            'left_hip_forward_swing_assessment': '',
            'right_hip_backward_swing_assessment': '',
            'right_hip_forward_swing_assessment': '',

            'vertical_oscillation': 0.0,
            'vertical_oscillation_assessment': '',
            
            'left_foot_strike': False,
            'right_foot_strike': False,
            'steps_per_minute': 0.0,
            'steps_per_minute_assessment': '',
            'recommendations': [],
        }
        self.angles = {}

    def calculate_metrics(self, valid_keypoints, timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.start_time is None:
            self.start_time = timestamp

        metrics = {metric: default_value for metric, default_value in self.key_metrics.items()}
        metrics['elapsed_time'] = timestamp - self.start_time

        self.angles = self.angle_metrics.calculate(valid_keypoints, metrics, timestamp)
        self.distance_metrics.calculate(valid_keypoints, metrics)

        metrics['recommendations'] = self.recommendations_calculator.get_recommendations(metrics)

        return metrics, self.angles

    def get_key_metrics(self) -> Dict[str, Any]:
        return self.key_metrics