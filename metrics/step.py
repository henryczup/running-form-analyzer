from typing import Dict, Literal
import numpy as np
from utils.foot_strike_detector import FootStrikeDetector
from utils.assessor import Assessor

class StepMetrics:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"], detection_axis: Literal["x", "y"]):
        self.left_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.right_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.left_step_count = 0
        self.right_step_count = 0
        self.total_step_count = 0
        self.last_left_knee_angle = 0.0
        self.last_right_knee_angle = 0.0
        self.last_left_hip_ankle_angle = 0.0
        self.last_right_hip_ankle_angle = 0.0

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        self.detect_foot_strikes(valid_keypoints, timestamp, metrics)
        self.calculate_step_counts(metrics)

    def detect_foot_strikes(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        self.detect_left_foot_strike(valid_keypoints, timestamp, metrics)
        self.detect_right_foot_strike(valid_keypoints, timestamp, metrics)

    def detect_left_foot_strike(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        if 6 in valid_keypoints:  # Left ankle
            metrics['left_foot_strike'], _ = self.left_foot_strike_detector.update(valid_keypoints[6], timestamp)
            if metrics['left_foot_strike']:
                self.left_step_count += 1
                self.update_assessments(metrics, left_is_front=True)
            metrics['left_ankle_position'] = self.left_foot_strike_detector.get_filtered_position()
            metrics['left_ankle_position_unfiltered'] = valid_keypoints[6][0]

    def detect_right_foot_strike(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        if 7 in valid_keypoints:  # Right ankle
            metrics['right_foot_strike'], _ = self.right_foot_strike_detector.update(valid_keypoints[7], timestamp)
            if metrics['right_foot_strike']:
                self.right_step_count += 1
                self.update_assessments(metrics, left_is_front=False)
            metrics['right_ankle_position'] = self.right_foot_strike_detector.get_filtered_position()
            metrics['right_ankle_position_unfiltered'] = valid_keypoints[7][0]

    def update_assessments(self, metrics: Dict[str, any], left_is_front: bool):
        if left_is_front:
            metrics['left_knee_assessment'] = Assessor.assess_knee_angle(self.last_left_knee_angle, is_front=True)
            metrics['right_knee_assessment'] = Assessor.assess_knee_angle(self.last_right_knee_angle, is_front=False)
            metrics['left_hip_ankle_angle_assessment'] = Assessor.assess_hip_ankle_angle(self.last_left_hip_ankle_angle)
        else:
            metrics['right_knee_assessment'] = Assessor.assess_knee_angle(self.last_right_knee_angle, is_front=True)
            metrics['left_knee_assessment'] = Assessor.assess_knee_angle(self.last_left_knee_angle, is_front=False)
            metrics['right_hip_ankle_angle_assessment'] = Assessor.assess_hip_ankle_angle(self.last_right_hip_ankle_angle)

    def calculate_step_counts(self, metrics: Dict[str, any]):
        self.total_step_count = self.left_step_count + self.right_step_count
        metrics['total_step_count'] = self.total_step_count
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60