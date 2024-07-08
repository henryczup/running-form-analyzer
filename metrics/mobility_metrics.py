from typing import Dict
from feedback.assessment_calculator import AssessmentCalculator

class MobilityMetrics:
    def __init__(self):
        self.max_left_backward_arm_swing = 0
        self.max_left_forward_arm_swing = 0
        self.max_right_backward_arm_swing = 0
        self.max_right_forward_arm_swing = 0
        self.max_left_backward_hip_swing = 0
        self.max_left_forward_hip_swing = 0
        self.max_right_backward_hip_swing = 0
        self.max_right_forward_hip_swing = 0
        self.prev_left_arm_angle = None
        self.prev_right_arm_angle = None
        self.prev_left_hip_angle = None
        self.prev_right_hip_angle = None

    def update(self, angles: Dict[str, float], metrics: Dict[str, any]):
        self.update_arm_swing_metrics(angles, metrics)
        self.update_hip_mobility_metrics(angles, metrics)

    def update_arm_swing_metrics(self, angles: Dict[str, float], metrics: Dict[str, any]):
        if 'left_arm_swing_angle' in angles:
            current_angle = angles['left_arm_swing_angle']
            if self.prev_left_arm_angle is not None:
                if current_angle > self.prev_left_arm_angle:  # Arm moving backward
                    self.max_left_backward_arm_swing = max(self.max_left_backward_arm_swing, current_angle)
                else:  # Arm moving forward
                    self.max_left_forward_arm_swing = max(self.max_left_forward_arm_swing, 180 - current_angle)
            self.prev_left_arm_angle = current_angle

        if 'right_arm_swing_angle' in angles:
            current_angle = angles['right_arm_swing_angle']
            if self.prev_right_arm_angle is not None:
                if current_angle > self.prev_right_arm_angle:  # Arm moving backward
                    self.max_right_backward_arm_swing = max(self.max_right_backward_arm_swing, current_angle)
                else:  # Arm moving forward
                    self.max_right_forward_arm_swing = max(self.max_right_forward_arm_swing, 180 - current_angle)
            self.prev_right_arm_angle = current_angle

        metrics['max_left_backward_arm_swing'] = self.max_left_backward_arm_swing
        metrics['max_left_forward_arm_swing'] = self.max_left_forward_arm_swing
        metrics['max_right_backward_arm_swing'] = self.max_right_backward_arm_swing
        metrics['max_right_forward_arm_swing'] = self.max_right_forward_arm_swing

        metrics['left_backward_arm_swing_assessment'] = AssessmentCalculator.assess_backward_arm_swing(self.max_left_backward_arm_swing)
        metrics['left_forward_arm_swing_assessment'] = AssessmentCalculator.assess_forward_arm_swing(self.max_left_forward_arm_swing)
        metrics['right_backward_arm_swing_assessment'] = AssessmentCalculator.assess_backward_arm_swing(self.max_right_backward_arm_swing)
        metrics['right_forward_arm_swing_assessment'] = AssessmentCalculator.assess_forward_arm_swing(self.max_right_forward_arm_swing)

    def update_hip_mobility_metrics(self, angles: Dict[str, float], metrics: Dict[str, any]):
        if 'left_hip_angle' in angles:
            current_angle = angles['left_hip_angle']
            if self.prev_left_hip_angle is not None:
                if current_angle > self.prev_left_hip_angle:  # Hip moving backward
                    self.max_left_backward_hip_swing = max(self.max_left_backward_hip_swing, current_angle - 90)
                else:  # Hip moving forward
                    self.max_left_forward_hip_swing = max(self.max_left_forward_hip_swing, 90 - current_angle)
            self.prev_left_hip_angle = current_angle

        if 'right_hip_angle' in angles:
            current_angle = angles['right_hip_angle']
            if self.prev_right_hip_angle is not None:
                if current_angle > self.prev_right_hip_angle:  # Hip moving backward
                    self.max_right_backward_hip_swing = max(self.max_right_backward_hip_swing, current_angle - 90)
                else:  # Hip moving forward
                    self.max_right_forward_hip_swing = max(self.max_right_forward_hip_swing, 90 - current_angle)
            self.prev_right_hip_angle = current_angle

        metrics['max_left_backward_hip_swing'] = self.max_left_backward_hip_swing
        metrics['max_left_forward_hip_swing'] = self.max_left_forward_hip_swing
        metrics['max_right_backward_hip_swing'] = self.max_right_backward_hip_swing
        metrics['max_right_forward_hip_swing'] = self.max_right_forward_hip_swing

        metrics['left_backward_hip_swing_assessment'] = AssessmentCalculator.assess_backward_hip_swing(self.max_left_backward_hip_swing)
        metrics['left_forward_hip_swing_assessment'] = AssessmentCalculator.assess_forward_hip_swing(self.max_left_forward_hip_swing)
        metrics['right_backward_hip_swing_assessment'] = AssessmentCalculator.assess_backward_hip_swing(self.max_right_backward_hip_swing)
        metrics['right_forward_hip_swing_assessment'] = AssessmentCalculator.assess_forward_hip_swing(self.max_right_forward_hip_swing)