from typing import Dict
import numpy as np
from utils.assessor import Assessor
from utils.angle_calculator import AngleCalculator

class AngleMetrics:
    def __init__(self):
        self.max_arm_swing_angle = 0
        self.last_arm_swing_direction = None
        self.arm_swing_threshold = 5  # degrees, adjust as needed

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any]):
        angles = AngleCalculator.calculate_all_angles(valid_keypoints)
        
        # Update metrics with calculated angles
        metrics.update(angles)

        # Perform assessments
        self.assess_angles(metrics)

        # Special handling for arm swing 
        self.handle_arm_swing(metrics)

    def assess_angles(self, metrics: Dict[str, any]):
        if 'head_angle' in metrics:
            metrics['head_angle_assessment'] = Assessor.assess_head_angle(metrics['head_angle'])

        if 'trunk_angle' in metrics:
            metrics['trunk_angle_assessment'] = Assessor.assess_torso_angle(abs(metrics['trunk_angle']))
        
        if 'left_elbow_angle' in metrics:
            metrics['left_elbow_angle_assessment'] = Assessor.assess_elbow_angle(metrics['left_elbow_angle'])

    # When the foot strikes detector becomes accurate,
    # remove this method and do these calculations in the step metrics class
    def handle_arm_swing(self, metrics: Dict[str, any]):
        if 'arm_swing_angle' in metrics:
            current_arm_swing_angle = metrics['arm_swing_angle']
            
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