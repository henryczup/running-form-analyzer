from typing import Dict
from feedback.assessment_calculator import AssessmentCalculator

class MobilityMetrics:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        self.limbs = ['left_arm', 'right_arm', 'left_hip', 'right_hip']
        for limb in self.limbs:
            setattr(self, f'{limb}_current_forward', 0)
            setattr(self, f'{limb}_current_backward', 0)
            setattr(self, f'{limb}_max_forward', 0)
            setattr(self, f'{limb}_max_backward', 0)
            setattr(self, f'{limb}_direction', 0)  # 0: neutral, 1: forward, -1: backward
            setattr(self, f'{limb}_prev_angle', 0)
            setattr(self, f'{limb}_forward_assessment', '')
            setattr(self, f'{limb}_backward_assessment', '')

    def update(self, angles: Dict[str, float], metrics: Dict[str, any]):
        self.update_swing_metrics(angles)
        self.update_assessments(metrics)

    def update_swing_metrics(self, angles: Dict[str, float]):
        if 'left_arm_swing_angle' in angles:
            self.process_swing('left_arm', angles['left_arm_swing_angle'])
        if 'right_arm_swing_angle' in angles:
            self.process_swing('right_arm', angles['right_arm_swing_angle'])
        if 'left_hip_angle' in angles:
            self.process_swing('left_hip', angles['left_hip_angle'])
        if 'right_hip_angle' in angles:
            self.process_swing('right_hip', angles['right_hip_angle'])

    def process_swing(self, limb: str, current_angle: float):
        prev_angle = getattr(self, f'{limb}_prev_angle')
        direction = getattr(self, f'{limb}_direction')

        if current_angle > prev_angle:
            if direction <= 0:  # Starting forward swing
                if direction == -1:  # Completed backward swing
                    self.finalize_swing(limb, 'backward')
                direction = 1
            if direction == 1:  # Continuing forward swing
                setattr(self, f'{limb}_current_forward', max(getattr(self, f'{limb}_current_forward'), current_angle))
        elif current_angle < prev_angle:
            if direction >= 0:  # Starting backward swing
                if direction == 1:  # Completed forward swing
                    self.finalize_swing(limb, 'forward')
                direction = -1
            if direction == -1:  # Continuing backward swing
                setattr(self, f'{limb}_current_backward', max(getattr(self, f'{limb}_current_backward'), abs(current_angle)))

        setattr(self, f'{limb}_prev_angle', current_angle)
        setattr(self, f'{limb}_direction', direction)

    def finalize_swing(self, limb: str, swing_type: str):
        current_max = getattr(self, f'{limb}_current_{swing_type}')
        if 'arm' in limb and current_max < 20:
            setattr(self, f'{limb}_current_{swing_type}', 0)
            return
        setattr(self, f'{limb}_max_{swing_type}', current_max)
        setattr(self, f'{limb}_current_{swing_type}', 0)

        if 'arm' in limb:
            assessment = (AssessmentCalculator.assess_forward_arm_swing(current_max) 
                          if swing_type == 'forward' 
                          else AssessmentCalculator.assess_backward_arm_swing(current_max))
        else:  # hip
            assessment = (AssessmentCalculator.assess_forward_hip_swing(current_max) 
                          if swing_type == 'forward' 
                          else AssessmentCalculator.assess_backward_hip_swing(current_max))
        
        setattr(self, f'{limb}_{swing_type}_assessment', assessment)

    def update_assessments(self, metrics: Dict[str, any]):
        for limb in self.limbs:
            metrics[f'max_{limb}_forward_swing'] = getattr(self, f'{limb}_max_forward')
            metrics[f'max_{limb}_backward_swing'] = getattr(self, f'{limb}_max_backward')
            metrics[f'{limb}_forward_swing_assessment'] = getattr(self, f'{limb}_forward_assessment')
            metrics[f'{limb}_backward_swing_assessment'] = getattr(self, f'{limb}_backward_assessment')