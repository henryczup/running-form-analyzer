from typing import Dict, Literal
import numpy as np
from feedback.assessment_calculator import AssessmentCalculator
from utils.foot_strike_detector import FootStrikeDetector

class StepMetrics:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"], detection_axis: Literal["x", "y"]):
        self.left_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.right_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.left_step_count = 0
        self.right_step_count = 0
        self.total_step_count = 0

        # Persistent assessments
        self.left_assessments = {
            'knee': '',
            'hip_ankle': '',
            'shank': ''
        }
        self.right_assessments = {
            'knee': '',
            'hip_ankle': '',
            'shank': ''
        }

    def update(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any], angles: Dict[str, float]):
        left_foot_strike, right_foot_strike = self.detect_foot_strikes(valid_keypoints, timestamp, metrics)
        self.calculate_step_counts(metrics)
        self.update_assessments(metrics, angles, left_foot_strike, right_foot_strike)

    def detect_foot_strikes(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, any]):
        left_foot_strike = False
        right_foot_strike = False

        if 15 in valid_keypoints:  # Left ankle
            left_foot_strike, _, _ = self.left_foot_strike_detector.update(valid_keypoints[15], timestamp)
            if left_foot_strike:
                self.left_step_count += 1

        if 16 in valid_keypoints:  # Right ankle
            right_foot_strike, _, _ = self.right_foot_strike_detector.update(valid_keypoints[16], timestamp)
            if right_foot_strike:
                self.right_step_count += 1

        metrics['left_foot_strike'] = left_foot_strike
        metrics['right_foot_strike'] = right_foot_strike

        return left_foot_strike, right_foot_strike

    def calculate_step_counts(self, metrics: Dict[str, any]):
        self.total_step_count = self.left_step_count + self.right_step_count
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60

    def update_assessments(self, metrics: Dict[str, any], angles: Dict[str, float], left_foot_strike: bool, right_foot_strike: bool):
        if left_foot_strike:
            if 'left_knee_angle' in angles:
                self.left_assessments['knee'] = AssessmentCalculator.assess_knee_angle(angles['left_knee_angle'], is_front=True)
            if 'left_hip_ankle_angle' in angles:
                self.left_assessments['hip_ankle'] = AssessmentCalculator.assess_hip_ankle_angle(angles['left_hip_ankle_angle'])
            if 'left_shank_angle' in angles:
                self.left_assessments['shank'] = AssessmentCalculator.assess_shank_angle(angles['left_shank_angle'])

        if right_foot_strike:
            if 'right_knee_angle' in angles:
                self.right_assessments['knee'] = AssessmentCalculator.assess_knee_angle(angles['right_knee_angle'], is_front=True)
            if 'right_hip_ankle_angle' in angles:
                self.right_assessments['hip_ankle'] = AssessmentCalculator.assess_hip_ankle_angle(angles['right_hip_ankle_angle'])
            if 'right_shank_angle' in angles:
                self.right_assessments['shank'] = AssessmentCalculator.assess_shank_angle(angles['right_shank_angle'])

        # Update metrics with the current assessments
        metrics['left_knee_assessment'] = self.left_assessments['knee']
        metrics['right_knee_assessment'] = self.right_assessments['knee']
        metrics['left_hip_ankle_angle_assessment'] = self.left_assessments['hip_ankle']
        metrics['right_hip_ankle_angle_assessment'] = self.right_assessments['hip_ankle']
        metrics['left_shank_angle_assessment'] = self.left_assessments['shank']
        metrics['right_shank_angle_assessment'] = self.right_assessments['shank']

        # Steps per minute assessment is updated on every frame
        metrics['steps_per_minute_assessment'] = AssessmentCalculator.assess_steps_per_minute(metrics['steps_per_minute'])