from collections import deque

class Recommendation:
    def __init__(self, window_size=30, consistency_threshold=0.7):
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        self.metric_history = {}
        self.recommendations = {
            'vertical_oscillation': "Reduce your vertical movement",
            'trunk_angle': "Adjust your torso angle",
            'left_elbow_angle': "Adjust your arm position",
            'left_knee_angle': "Adjust your left knee position",
            'right_knee_angle': "Adjust your right knee position",
            'left_arm_swing_angle': "Adjust your arm swing",
            'left_hip_ankle_angle': "Adjust your left leg position",
            'right_hip_ankle_angle': "Adjust your right leg position"
        }

    def update(self, metrics):
        for metric, assessment in metrics.items():
            if metric not in self.metric_history:
                self.metric_history[metric] = deque(maxlen=self.window_size)
            if assessment:  # Only add non-empty assessments
                self.metric_history[metric].append(assessment)

    def needs_improvement(self, metric):
        if metric not in self.metric_history:
            return False
        history = self.metric_history[metric]
        if not history:  # If history is empty, return False
            return False
        
        # Count both 'Bad' and 'Need Improvement' assessments
        count = sum(1 for assessment in history if assessment in ['Bad', 'Need Improvement'])
        return count / len(history) >= self.consistency_threshold

    def any_metric_needs_improvement(self):
        return any(self.needs_improvement(metric) for metric in self.recommendations.keys())

    def get_recommendations(self, metrics):
        self.update({
            'vertical_oscillation': metrics.get('vertical_oscillation_assessment'),
            'trunk_angle': metrics.get('trunk_angle_assessment'),
            'left_elbow_angle': metrics.get('left_elbow_angle_assessment'),
            'left_knee_angle': metrics.get('left_knee_assessment'),
            'right_knee_angle': metrics.get('right_knee_assessment'),
            'left_arm_swing_angle': metrics.get('left_arm_swing_angle_assessment'),
            'left_hip_ankle_angle': metrics.get('left_hip_ankle_angle_assessment'),
            'right_hip_ankle_angle': metrics.get('right_hip_ankle_angle_assessment')
        })

        if not self.any_metric_needs_improvement():
            return []

        return [self.recommendations[metric] for metric in self.recommendations if self.needs_improvement(metric)]
