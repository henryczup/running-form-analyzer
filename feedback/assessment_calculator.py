class AssessmentCalculator:
    @staticmethod
    def assess_head_angle(angle: float) -> str:
        if angle < 80:
            return 'Bad'
        elif 80 <= angle < 85:
            return 'Need Improvement'
        elif 85 <= angle <= 95:
            return 'Good'
        elif 95 < angle <= 100:
            return 'Need Improvement'
        else:
            return 'Bad'
        
    @staticmethod
    def assess_knee_angle(angle: float, is_front: bool) -> str:
        if is_front:
            if angle > 135:
                return 'Good'
            elif 126 <= angle <= 135:
                return 'Need Improvement'
            else:
                return 'Bad'
        else:  # back knee
            if angle < 64:
                return 'Good'
            elif 64 <= angle <= 73:
                return 'Need Improvement'
            else:
                return 'Bad'
            
    @staticmethod
    def assess_torso_angle(angle: float) -> str:
        if angle < 1:
            return 'Bad'
        elif 1 <= angle < 5:
            return 'Need Improvement'
        elif 5 <= angle <= 11:
            return 'Good'
        elif 11 < angle <= 15:
            return 'Need Improvement'
        else:
            return 'Bad'

    @staticmethod
    def assess_elbow_angle(angle: float) -> str:
        if angle < 77:
            return 'Bad'
        elif 77 <= angle < 79:
            return 'Need Improvement'
        elif 79 <= angle <= 89:
            return 'Good'
        elif 89 < angle <= 93:
            return 'Need Improvement'
        else:
            return 'Bad'

    @staticmethod
    def assess_backward_arm_swing(angle: float) -> str:
        if angle < 20:
            return 'Bad'
        elif 20 <= angle < 30:
            return 'Need Improvement'
        elif 30 <= angle <= 45:
            return 'Good'
        elif 45 < angle <= 55:
            return 'Need Improvement'
        else:
            return 'Bad'

    @staticmethod
    def assess_forward_arm_swing(angle: float) -> str:
        if angle < 15:
            return 'Bad'
        elif 15 <= angle < 25:
            return 'Need Improvement'
        elif 25 <= angle <= 40:
            return 'Good'
        elif 40 < angle <= 50:
            return 'Need Improvement'
        else:
            return 'Bad'
    @staticmethod
    def assess_vertical_oscillation(oscillation: float) -> str:
        if oscillation < 6.5:
            return 'Good'
        elif 6.5 <= oscillation < 8:
            return 'Need Improvement'
        else:
            return 'Bad'

    @staticmethod
    def assess_hip_ankle_angle(angle: float) -> str:
        if angle < 145:
            return 'Bad'
        elif 145 <= angle < 150:
            return 'Need Improvement'
        elif 150 <= angle <= 160:
            return 'Good'
        elif 160 < angle <= 165:
            return 'Need Improvement'
        else:
            return 'Bad'
        
    @staticmethod
    def assess_steps_per_minute(steps_per_minute: float) -> str:
        if steps_per_minute < 155:
            return 'Bad'
        elif 155 <= steps_per_minute < 165:
            return 'Need Improvement'
        elif 165 <= steps_per_minute <= 180:
            return 'Good'
        elif 180 < steps_per_minute <= 190:
            return 'Need Improvement'
        else:
            return 'Bad'
        
    @staticmethod
    def assess_backward_hip_swing(angle: float) -> str:
        if angle < 10:
            return 'Bad'
        elif 10 <= angle < 15:
            return 'Need Improvement'
        else:
            return 'Good'

    @staticmethod
    def assess_forward_hip_swing(angle: float) -> str:
        if angle < 30:
            return 'Bad'
        elif 30 <= angle < 40:
            return 'Need Improvement'
        elif 40 <= angle <= 50:
            return 'Good'

    @staticmethod
    def assess_shank_angle(angle: float) -> str:
        if 0 <= angle <= 5:
            return 'Good'
        elif 5 < angle <= 10:
            return 'Need Improvement'
        else:
            return 'Bad'