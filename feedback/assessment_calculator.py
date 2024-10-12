class AssessmentCalculator:
    @staticmethod
    def assess_head_angle(angle: float) -> str:
        if angle < 65:
            return 'Bad'
        elif 65 <= angle < 80:
            return 'Need Improvement'
        elif 80 <= angle <= 110:
            return 'Good'
        elif 110 < angle <= 120:
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
        elif 5 <= angle <= 15:
            return 'Good'
        elif 15 < angle <= 17:
            return 'Need Improvement'
        else:
            return 'Bad'

    @staticmethod
    def assess_elbow_angle(angle: float) -> str:
        if angle < 53:
            return 'Bad'
        elif 53 <= angle < 60:
            return 'Need Improvement'
        elif 60 <= angle <= 90:
            return 'Good'
        elif 89 < angle <= 95:
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
        if angle < 35:
            return 'Bad'
        elif 35 <= angle < 45:
            return 'Need Improvement'
        elif 45 <= angle <= 65:
            return 'Good'
        elif 65 < angle <= 75:
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
        # if angle < :
        #     return 'Bad'
        # elif  <= angle < :
        #     return 'Need Improvement'
        if 0 <= angle <= 15:
            return 'Good'
        elif 15 < angle <= 20:
            return 'Need Improvement'
        else:
            return 'Bad'
        
    @staticmethod
    def assess_backward_hip_swing(angle: float) -> str:
        if angle < 29:
            return 'Bad'
        elif 29 <= angle < 41:
            return 'Good'
        else:
            return 'Need Improvement'

    @staticmethod
    def assess_forward_hip_swing(angle: float) -> str:
        if angle < 29:
            return 'Bad'
        elif 29 <= angle < 41:
            return 'Good'
        else:
            return 'Need Improvement'

    @staticmethod
    def assess_shank_angle(angle: float) -> str:
        if 0 <= angle <= 10:
            return 'Good'
        elif 10 < angle <= 15:
            return 'Need Improvement'
        else:
            return 'Bad'