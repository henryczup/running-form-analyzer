from collections import deque

from feedback.audio_feedback import AudioFeedbackProvider

class Recommendation:
    def __init__(self, window_size=30, consistency_threshold=0.7, audio_provider: AudioFeedbackProvider = None):
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        self.metric_history = {}
        self.audio_provider = audio_provider 
        self.recommendations = {
            'head_angle': 'Adjust your head position',
            'trunk_angle': 'Adjust your torso position',
            'left_elbow_angle': 'Adjust your left elbow angle',
            'right_elbow_angle': 'Adjust your right elbow angle',
            'left_hip_ankle_angle_at_strike': 'Adjust your left foot strike',
            'right_hip_ankle_angle_at_strike': 'Adjust your right foot strike',
            'left_knee_angle_at_strike': 'Adjust your left knee position',
            'right_knee_angle_at_strike': 'Adjust your right knee position',
            'left_shank_angle_at_strike': 'Adjust your left shank position',
            'right_shank_angle_at_strike': 'Adjust your right shank position',
            'max_left_arm_backward_swing': 'Adjust your left arm backward swing',
            'max_left_arm_forward_swing': 'Adjust your left arm forward swing',
            'max_right_arm_backward_swing': 'Adjust your right arm backward swing',
            'max_right_arm_forward_swing': 'Adjust your right arm forward swing',
            'max_left_hip_backward_swing': 'Adjust your left hip backward swing',
            'max_left_hip_forward_swing': 'Adjust your left hip forward swing',
            'max_right_hip_backward_swing': 'Adjust your right hip backward swing',
            'max_right_hip_forward_swing': 'Adjust your right hip forward swing',
            'vertical_oscillation': 'Adjust your vertical oscillation',
            'steps_per_minute': 'Adjust your steps per minute'
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


    def get_recommendations(self, metrics):
        self.update(metrics)
        recommendations = [self.recommendations[metric] for metric in self.recommendations if self.needs_improvement(metric)]
        
        if self.audio_provider:
            for recommendation in recommendations:
                self.audio_provider.add_feedback(recommendation)
        
        return recommendations
