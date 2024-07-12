import numpy as np
from typing import Dict

from core.config import Config

class AngleCalculator:
    @staticmethod
    def calculate_angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        return np.degrees(np.arccos(dot_product / magnitudes))

    @staticmethod
    def calculate_all_angles(valid_keypoints: Dict[int, np.ndarray], angles: Dict[str, float], config: Config) -> Dict[str, float]:

 # Head angle
        if config.side == 'left':
            if all(i in valid_keypoints for i in [1, 3, 5, 11]):  # Left eye, left ear, left shoulder, left hip
                mid_ear = valid_keypoints[3]
                mid_eye = valid_keypoints[1]
                shoulder = valid_keypoints[5]
                hip = valid_keypoints[11]
                body_vector = hip - shoulder
                head_vector = mid_eye - mid_ear
                angles['head_angle'] = AngleCalculator.calculate_angle(body_vector, head_vector)
        else:  # 'right' side
            if all(i in valid_keypoints for i in [2, 4, 6, 12]):  # Right eye, right ear, right shoulder, right hip
                mid_ear = valid_keypoints[4]
                mid_eye = valid_keypoints[2]
                shoulder = valid_keypoints[6]
                hip = valid_keypoints[12]
                body_vector = hip - shoulder
                head_vector = mid_eye - mid_ear
                angles['head_angle'] = AngleCalculator.calculate_angle(body_vector, head_vector)

        # Trunk angle
        if config.side == 'left':
            if all(i in valid_keypoints for i in [5, 11]):  # Left shoulder, left hip
                shoulder = valid_keypoints[5]
                hip = valid_keypoints[11]
                trunk_line = shoulder - hip
                vertical_line = np.array([0, -1])
                angles['trunk_angle'] = AngleCalculator.calculate_angle(trunk_line, vertical_line)
                angles['trunk_angle'] = -angles['trunk_angle'] if trunk_line[0] < 0 else angles['trunk_angle']
        else:  # 'right' side
            if all(i in valid_keypoints for i in [6, 12]):  # Right shoulder, right hip
                shoulder = valid_keypoints[6]
                hip = valid_keypoints[12]
                trunk_line = shoulder - hip
                vertical_line = np.array([0, -1])
                angles['trunk_angle'] = AngleCalculator.calculate_angle(trunk_line, vertical_line)
                angles['trunk_angle'] = -angles['trunk_angle'] if trunk_line[0] < 0 else angles['trunk_angle']

                # Elbow angle
        if config.side == 'left':
            if all(i in valid_keypoints for i in [5, 7, 9]):  # Left shoulder, left elbow, left wrist
                upper_arm = valid_keypoints[7] - valid_keypoints[5]  # shoulder - elbow
                forearm = valid_keypoints[7] - valid_keypoints[9]  # wrist - elbow
                angles['left_elbow_angle'] = AngleCalculator.calculate_angle(upper_arm, forearm)
        else:  # 'right' side
            if all(i in valid_keypoints for i in [6, 8, 10]):  # Right shoulder, right elbow, right wrist
                upper_arm = valid_keypoints[8] - valid_keypoints[6]  # shoulder - elbow
                forearm = valid_keypoints[8] - valid_keypoints[10]  # wrist - elbow
                angles['right_elbow_angle'] = AngleCalculator.calculate_angle(upper_arm, forearm)

        # arm swing angle
        if config.side == 'left':
            if all(i in valid_keypoints for i in [5, 7, 11]):
                torso_vector = valid_keypoints[5]- valid_keypoints[11]
                upper_arm_vector = valid_keypoints[5] - valid_keypoints[7]
                angles['left_arm_swing_angle'] = AngleCalculator.calculate_angle(torso_vector, upper_arm_vector)
                angles['left_arm_swing_angle'] = -angles['left_arm_swing_angle'] if upper_arm_vector[0] < 0 else angles['left_arm_swing_angle']
        else:  # 'right' side
            if all(i in valid_keypoints for i in [6, 8, 12]):
                torso_vector = valid_keypoints[6] -  valid_keypoints[12]
                upper_arm_vector = valid_keypoints[6] - valid_keypoints[8]
                angles['right_arm_swing_angle'] = AngleCalculator.calculate_angle(torso_vector, upper_arm_vector)
                angles['right_arm_swing_angle'] = -angles['right_arm_swing_angle'] if upper_arm_vector[0] < 0 else angles['right_arm_swing_angle']

        # Left knee angle
        if all(i in valid_keypoints for i in [11, 13, 15]):  # Left hip, left knee, left ankle
            thigh = valid_keypoints[13] - valid_keypoints[11] 
            lower_leg = valid_keypoints[13]- valid_keypoints[15]
            angles['left_knee_angle'] = AngleCalculator.calculate_angle(thigh, lower_leg)

        # Right knee angle
        if all(i in valid_keypoints for i in [12, 14, 16]):  # Right hip, right knee, right ankle
            thigh = valid_keypoints[14]- valid_keypoints[12]
            lower_leg = valid_keypoints[14] - valid_keypoints[16]
            angles['right_knee_angle'] = AngleCalculator.calculate_angle(thigh, lower_leg)

        # Left hip angle
        if all(i in valid_keypoints for i in [5, 11, 13]):  # Left shoulder, left hip, left knee
            torso_vector = valid_keypoints[11] - valid_keypoints[5]  # Left hip - Left shoulder
            thigh_vector = valid_keypoints[13] - valid_keypoints[11]  # Left knee - Left hip
            angles['left_hip_angle'] = AngleCalculator.calculate_angle(torso_vector, thigh_vector)
            angles['left_hip_angle'] = -angles['left_hip_angle'] if thigh_vector[0] < 0 else angles['left_hip_angle']

        # Right hip angle
        if all(i in valid_keypoints for i in [6, 12, 14]):  # Right shoulder, right hip, right knee
            torso_vector = valid_keypoints[12] - valid_keypoints[6]  # Right hip - Right shoulder
            thigh_vector = valid_keypoints[14] - valid_keypoints[12]  # Right knee - Right hip
            angles['right_hip_angle'] = AngleCalculator.calculate_angle(torso_vector, thigh_vector)
            angles['right_hip_angle'] = -angles['right_hip_angle'] if thigh_vector[0] < 0 else angles['right_hip_angle']

        # Hip-to-ankle angles
        if all(i in valid_keypoints for i in [11, 15]):  # Left hip, left ankle
            hip_ankle_line = valid_keypoints[15] - valid_keypoints[11]
            vertical_line = np.array([0, 1])
            angles['left_hip_ankle_angle'] = AngleCalculator.calculate_angle(hip_ankle_line, vertical_line)

        if all(i in valid_keypoints for i in [12, 16]):  # Right hip, right ankle
            hip_ankle_line = valid_keypoints[16] - valid_keypoints[12]
            vertical_line = np.array([0, 1])
            angles['right_hip_ankle_angle'] = AngleCalculator.calculate_angle(hip_ankle_line, vertical_line)

         # Shank angle calculations
        if all(i in valid_keypoints for i in [13, 15]):  # Left knee, left ankle
            horizontal_vector = np.array([0, 1])
            left_shin_vector = valid_keypoints[15] - valid_keypoints[13]
            angles['left_shank_angle'] = AngleCalculator.calculate_angle(horizontal_vector, left_shin_vector)

        if all(i in valid_keypoints for i in [14, 16]):  # Right knee, right ankle
            horizontal_vector = np.array([0, 1])
            right_shin_vector = valid_keypoints[16] - valid_keypoints[14]
            angles['right_shank_angle'] = AngleCalculator.calculate_angle(horizontal_vector, right_shin_vector)


        return angles