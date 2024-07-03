import numpy as np
from typing import Dict

class AngleCalculator:
    @staticmethod
    def calculate_angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return np.degrees(np.arccos(dot_product / magnitudes))

    @staticmethod
    def calculate_all_angles(valid_keypoints: Dict[int, np.ndarray]) -> Dict[str, float]:
        angles = {}

        # Head angle
        if all(i in valid_keypoints for i in [0, 1, 2]):  # Assuming 0: nose, 1: left eye, 2: right eye
            eye_midpoint = (valid_keypoints[1] + valid_keypoints[2]) / 2
            head_vector = valid_keypoints[0] - eye_midpoint
            vertical_vector = np.array([1, 0])  # Pointing left
            angles['head_angle'] = AngleCalculator.calculate_angle(head_vector, vertical_vector)
            angles['head_angle'] = -angles['head_angle'] if head_vector[0] < 0 else angles['head_angle']

        # Trunk angle
        if all(i in valid_keypoints for i in [0, 1, 2, 3]):
            shoulder_midpoint = (valid_keypoints[0] + valid_keypoints[1]) / 2
            hip_midpoint = (valid_keypoints[2] + valid_keypoints[3]) / 2
            trunk_line = hip_midpoint - shoulder_midpoint
            vertical_line = np.array([0, 1])
            angles['trunk_angle'] = AngleCalculator.calculate_angle(trunk_line, vertical_line)
            angles['trunk_angle'] = -angles['trunk_angle'] if trunk_line[0] < 0 else angles['trunk_angle']

        # Knee angles
        if all(i in valid_keypoints for i in [2, 4, 6]):
            thigh = valid_keypoints[2] - valid_keypoints[4]
            lower_leg = valid_keypoints[6] - valid_keypoints[4]
            angles['left_knee_angle'] = 180 - AngleCalculator.calculate_angle(thigh, lower_leg)
        if all(i in valid_keypoints for i in [3, 5, 7]):
            thigh = valid_keypoints[3] - valid_keypoints[5]
            lower_leg = valid_keypoints[7] - valid_keypoints[5]
            angles['right_knee_angle'] = 180 - AngleCalculator.calculate_angle(thigh, lower_leg)

        # Elbow angle
        if all(i in valid_keypoints for i in [5, 7, 9]):
            upper_arm = valid_keypoints[7] - valid_keypoints[5]
            forearm = valid_keypoints[9] - valid_keypoints[7]
            angles['left_elbow_angle'] = 180 - AngleCalculator.calculate_angle(upper_arm, forearm)

        # Arm swing angle
        if all(i in valid_keypoints for i in [0, 8]):
            upper_arm = valid_keypoints[8] - valid_keypoints[0]
            vertical_line = np.array([0, 1])
            angles['arm_swing_angle'] = AngleCalculator.calculate_angle(upper_arm, vertical_line)

        # Hip-ankle angles
        if all(i in valid_keypoints for i in [2, 6]):
            hip_ankle_line = valid_keypoints[6] - valid_keypoints[2]
            vertical_line = np.array([0, 1])
            angles['left_hip_ankle_angle'] = AngleCalculator.calculate_angle(hip_ankle_line, vertical_line)
        if all(i in valid_keypoints for i in [3, 7]):
            hip_ankle_line = valid_keypoints[7] - valid_keypoints[3]
            vertical_line = np.array([0, 1])
            angles['right_hip_ankle_angle'] = AngleCalculator.calculate_angle(hip_ankle_line, vertical_line)

        return angles