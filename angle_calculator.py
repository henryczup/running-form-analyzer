import numpy as np


class AngleCalculator:
    @staticmethod
    def calculate_angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return np.degrees(np.arccos(dot_product / magnitudes))

    @staticmethod
    def calculate_trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
        trunk_line = hip - shoulder
        vertical_line = np.array([0, 1])
        angle = AngleCalculator.calculate_angle(trunk_line, vertical_line)
        return -angle if trunk_line[0] < 0 else angle

    @staticmethod
    def calculate_knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        thigh = hip - knee
        lower_leg = ankle - knee
        return 180 - AngleCalculator.calculate_angle(thigh, lower_leg)

    @staticmethod
    def calculate_arm_swing_angle(shoulder: np.ndarray, elbow: np.ndarray) -> float:
        upper_arm = elbow - shoulder
        vertical_line = np.array([0, 1])
        return AngleCalculator.calculate_angle(upper_arm, vertical_line)

    @staticmethod
    def calculate_hip_ankle_angle(hip: np.ndarray, ankle: np.ndarray) -> float:
        hip_ankle_line = ankle - hip
        vertical_line = np.array([0, 1])
        return AngleCalculator.calculate_angle(hip_ankle_line, vertical_line)