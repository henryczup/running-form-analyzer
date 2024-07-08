from typing import Dict, List
import numpy as np
from collections import deque
from core.config import HFOV_DEG, IMAGE_WIDTH_PX, KNOWN_TORSO_LENGTH_CM
from feedback.assessment_calculator import AssessmentCalculator

HFOV_RAD = np.radians(HFOV_DEG)
# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)

class DistanceMetrics:
    def __init__(self):
        self.hip_positions = deque(maxlen=10)
        self.current_distance = 0.0

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        self.update_distance(valid_keypoints)
        self.calculate_vertical_oscillation(metrics)

    def update_distance(self, valid_keypoints: Dict[int, np.ndarray]):
        if all(i in valid_keypoints for i in [0, 1, 2, 3]):
            shoulder_midpoint = (valid_keypoints[0] + valid_keypoints[1]) / 2
            hip_midpoint = (valid_keypoints[2] + valid_keypoints[3]) / 2
            torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
            self.current_distance = (KNOWN_TORSO_LENGTH_CM * FOCAL_LENGTH_PX) / torso_length_px
            
            self.hip_positions.append(hip_midpoint[1])

    def calculate_vertical_oscillation(self, metrics: Dict[str, any]):
        if len(self.hip_positions) < 2:
            metrics['vertical_oscillation'] = 0.0
        else:
            moving_avg = np.convolve(list(self.hip_positions), np.ones(5), 'valid') / 5
            oscillation = (np.max(moving_avg) - np.min(moving_avg)) / 2
            metrics['vertical_oscillation'] = oscillation * (self.current_distance / FOCAL_LENGTH_PX)
        
        metrics['vertical_oscillation_assessment'] = AssessmentCalculator.assess_vertical_oscillation(metrics['vertical_oscillation'])

    def get_current_distance(self) -> float:
        return self.current_distance