from typing import Dict
import numpy as np
from utils.angle_calculator import AngleCalculator
from utils.assessor import Assessor

class Angle:
    def __init__(self):
        self.max_arm_swing_angle = 0
        self.last_arm_swing_direction = None
        self.arm_swing_threshold = 5  # degrees, adjust as needed

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        self.calculate_trunk_angle(valid_keypoints, metrics)
        self.calculate_elbow_angle(valid_keypoints, metrics)
        self.calculate_knee_angles(valid_keypoints, metrics)
        self.calculate_arm_swing_angle(valid_keypoints, metrics)
        self.calculate_hip_ankle_angles(valid_keypoints, metrics)

    def calculate_trunk_angle(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        if all(i in valid_keypoints for i in [0, 1, 2, 3]):
            shoulder_midpoint = (valid_keypoints[0] + valid_keypoints[1]) / 2
            hip_midpoint = (valid_keypoints[2] + valid_keypoints[3]) / 2
            metrics['trunk_angle'] = AngleCalculator.calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
            metrics['trunk_angle_assessment'] = Assessor.assess_torso_angle(abs(metrics['trunk_angle']))

    def calculate_elbow_angle(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        if all(i in valid_keypoints for i in [5, 7, 9]):
            metrics['left_elbow_angle'] = AngleCalculator.calculate_elbow_angle(
                valid_keypoints[5], valid_keypoints[7], valid_keypoints[9]
            )
            metrics['left_elbow_angle_assessment'] = Assessor.assess_elbow_angle(metrics['left_elbow_angle'])

    def calculate_knee_angles(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        if all(i in valid_keypoints for i in [2, 4, 6]):
            metrics['left_knee_angle'] = AngleCalculator.calculate_knee_angle(
                valid_keypoints[2], valid_keypoints[4], valid_keypoints[6])
        if all(i in valid_keypoints for i in [3, 5, 7]):
            metrics['right_knee_angle'] = AngleCalculator.calculate_knee_angle(
                valid_keypoints[3], valid_keypoints[5], valid_keypoints[7])

    def calculate_arm_swing_angle(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        if all(i in valid_keypoints for i in [0, 8]):
            current_arm_swing_angle = AngleCalculator.calculate_arm_swing_angle(
                valid_keypoints[0], valid_keypoints[8])
            
            if self.last_arm_swing_direction is None:
                self.last_arm_swing_direction = current_arm_swing_angle > self.max_arm_swing_angle
            else:
                current_direction = current_arm_swing_angle > self.max_arm_swing_angle
                if current_direction != self.last_arm_swing_direction:
                    metrics['max_arm_swing_angle'] = self.max_arm_swing_angle
                    metrics['max_arm_swing_angle_assessment'] = Assessor.assess_arm_swing_angle(self.max_arm_swing_angle)
                    self.max_arm_swing_angle = current_arm_swing_angle
                self.last_arm_swing_direction = current_direction
            
            if abs(current_arm_swing_angle - self.max_arm_swing_angle) > self.arm_swing_threshold:
                self.max_arm_swing_angle = max(self.max_arm_swing_angle, current_arm_swing_angle)
            
            metrics['arm_swing_angle'] = current_arm_swing_angle

    def calculate_hip_ankle_angles(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        if all(i in valid_keypoints for i in [2, 6]):
            metrics['left_hip_ankle_angle'] = AngleCalculator.calculate_hip_ankle_angle(
                valid_keypoints[2], valid_keypoints[6])
        if all(i in valid_keypoints for i in [3, 7]):
            metrics['right_hip_ankle_angle'] = AngleCalculator.calculate_hip_ankle_angle(
                valid_keypoints[3], valid_keypoints[7])