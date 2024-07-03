from typing import Dict, Literal
import numpy as np
from utils.assessor import Assessor
from utils.foot_strike_detector import FootStrikeDetector
from utils.angle_calculator import AngleCalculator

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
        self.left_knee_assessment = ''
        self.right_knee_assessment = ''
        self.left_hip_ankle_assessment = ''
        self.right_hip_ankle_assessment = ''

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        angles = AngleCalculator.calculate_all_angles(valid_keypoints)
        metrics.update(angles)

        self.detect_foot_strikes(valid_keypoints, timestamp, metrics)
        self.calculate_step_counts(metrics)
        self.update_assessments(metrics)

    def detect_foot_strikes(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        if 6 in valid_keypoints:  # Left ankle
            metrics['left_foot_strike'], _ = self.left_foot_strike_detector.update(valid_keypoints[6], timestamp)
            if metrics['left_foot_strike']:
                self.left_step_count += 1
                self.update_left_assessments(metrics)
            metrics['left_ankle_position'] = self.left_foot_strike_detector.get_filtered_position()

        if 7 in valid_keypoints:  # Right ankle
            metrics['right_foot_strike'], _ = self.right_foot_strike_detector.update(valid_keypoints[7], timestamp)
            if metrics['right_foot_strike']:
                self.right_step_count += 1
                self.update_right_assessments(metrics)
            metrics['right_ankle_position'] = self.right_foot_strike_detector.get_filtered_position()

    def update_left_assessments(self, metrics: Dict[str, any]):
        if 'left_knee_angle' in metrics:
            self.left_knee_assessment = Assessor.assess_knee_angle(metrics['left_knee_angle'], is_front=True)
        if 'left_hip_ankle_angle' in metrics:
            self.left_hip_ankle_assessment = Assessor.assess_hip_ankle_angle(metrics['left_hip_ankle_angle'])

    def update_right_assessments(self, metrics: Dict[str, any]):
        if 'right_knee_angle' in metrics:
            self.right_knee_assessment = Assessor.assess_knee_angle(metrics['right_knee_angle'], is_front=True)
        if 'right_hip_ankle_angle' in metrics:
            self.right_hip_ankle_assessment = Assessor.assess_hip_ankle_angle(metrics['right_hip_ankle_angle'])

    def calculate_step_counts(self, metrics: Dict[str, any]):
        self.total_step_count = self.left_step_count + self.right_step_count
        metrics['total_step_count'] = self.total_step_count
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60

    def update_assessments(self, metrics: Dict[str, any]):
        metrics['left_knee_assessment'] = self.left_knee_assessment
        metrics['right_knee_assessment'] = self.right_knee_assessment
        metrics['left_hip_ankle_angle_assessment'] = self.left_hip_ankle_assessment
        metrics['right_hip_ankle_angle_assessment'] = self.right_hip_ankle_assessment