class Assessor:
    @staticmethod
    def assess_head_angle(angle: float) -> str:
        if angle < -16:
            return 'Bad'
        elif -16 <= angle < -12:
            return 'Need Improvement'
        elif -12 <= angle <= -2:
            return 'Good'
        elif -2 < angle <= 2:
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
    def assess_arm_swing_angle(angle: float) -> str:
        if angle < 40:
            return 'Bad'
        elif 40 <= angle < 50:
            return 'Need Improvement'
        elif 50 <= angle <= 70:
            return 'Good'
        elif 70 < angle <= 80:
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