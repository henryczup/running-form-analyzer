from typing import Dict
import numpy as np
from core.config import Config
from metrics.mobility_metrics import MobilityMetrics
from metrics.step_metrics import StepMetrics
from utils.angle_calculator import AngleCalculator
from feedback.assessment_calculator import AssessmentCalculator

class AngleMetrics:
    def __init__(self, config: Config):
        self.config = config
        self.angles = {}
        self.mobility_metrics = MobilityMetrics()
        self.step_metrics = StepMetrics(filter_type='temporal', detection_axis='y')

    def calculate(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, any], timestamp: float):
        self.angles = AngleCalculator.calculate_all_angles(valid_keypoints, self.angles, self.config)
        metrics.update(self.angles)

        self.update_static_angles_assessment(metrics)

        self.mobility_metrics.update(self.angles, metrics)

        self.step_metrics.update(valid_keypoints, timestamp, metrics, self.angles)

        return self.angles

    def update_static_angles_assessment(self, metrics: Dict[str, any]):
        if 'head_angle' in metrics:
            metrics['head_angle_assessment'] = AssessmentCalculator.assess_head_angle(metrics['head_angle'])

        if 'trunk_angle' in metrics:
            metrics['trunk_angle_assessment'] = AssessmentCalculator.assess_torso_angle(abs(metrics['trunk_angle']))
        
        if self.config.side == 'left':
            if 'left_elbow_angle' in metrics:
                metrics['left_elbow_angle_assessment'] = AssessmentCalculator.assess_elbow_angle(metrics['left_elbow_angle'])
        else:
            if 'right_elbow_angle' in metrics:
                metrics['right_elbow_angle_assessment'] = AssessmentCalculator.assess_elbow_angle(metrics['right_elbow_angle'])