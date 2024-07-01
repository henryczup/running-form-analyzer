
# Convert HFOV to radians
from collections import deque
from typing import Any, Dict, List, Literal, Tuple
import numpy as np

from display import display_dev_mode, display_user_mode
from angle_calculator import AngleCalculator
from assessor import Assessor
from config import HFOV_DEG, IMAGE_WIDTH_PX, KNOWN_TORSO_LENGTH_CM
from foot_strike_detector import FootStrikeDetector
from recommendations_calculator import RecommendationsCalculator


HFOV_RAD = np.radians(HFOV_DEG)
# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)

        

class MetricsCalculator:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "temporal",
                 detection_axis: Literal["x", "y"] = "y"):
        self.hip_positions = deque(maxlen=10)
        self.left_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.right_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.recommendations_calculator = RecommendationsCalculator(window_size=30, consistency_threshold=0.7)
        self.left_step_count = 0
        self.right_step_count = 0
        self.total_step_count = 0
        self.start_time = None

        self.max_arm_swing_angle = 0
        self.last_arm_swing_direction = None
        self.arm_swing_threshold = 5  # degrees, adjust as needed

        # TODO: change to make them more modular later
        self.last_left_knee_angle = 0.0
        self.last_right_knee_angle = 0.0
        self.last_left_hip_ankle_angle = 0.0
        self.last_right_hip_ankle_angle = 0.0

        self.available_metrics = {
            'trunk_angle': 0.0,
            'trunk_angle_assessment': '',

            'left_knee_angle': 0.0,
            'right_knee_angle': 0.0,
            'left_knee_assessment': '',
            'right_knee_assessment': '',

            'max_left_arm_swing_angle': 0.0,
            'max_left_arm_swing_angle_assessment': '',

            'left_elbow_angle': 0.0,
            'left_elbow_angle_assessment': '',

            'distance_cm': 0.0,
            'vertical_oscillation': 0.0,
            'vertical_oscillation_assessment': '',

            'left_hip_ankle_angle': 0.0,
            'right_hip_ankle_angle': 0.0,
            'left_hip_ankle_angle_assessment': '',
            'right_hip_ankle_angle_assessment': '',

            'left_foot_strike': False,
            'right_foot_strike': False,
            'left_ankle_position': 0.0,
            'right_ankle_position': 0.0,
            
            'total_step_count': 0,
            'steps_per_minute': 0.0,
            'elapsed_time': 0.0,

            'recommendations': [],
        }

    
    def calculate_metrics(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], frame: np.ndarray, 
                          confidence_threshold: float, timestamp: float, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate various metrics based on keypoint data."""
        if self.start_time is None:
            self.start_time = timestamp

        metrics = {metric: default_value for metric, default_value in self.available_metrics.items()}

        metrics['elapsed_time'] = timestamp - self.start_time

        self.calculate_body_angles(keypoint_coords, keypoint_confs, confidence_threshold, metrics)
        self.detect_foot_strikes(keypoint_coords, keypoint_confs, confidence_threshold, timestamp, metrics)
        self.calculate_step_metrics(metrics)        
            
        # Calculate total step count
        self.total_step_count = self.left_step_count + self.right_step_count
        metrics['total_step_count'] = self.total_step_count

        # Calculate steps per minute
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60

        # Calculate recommendations
        metrics['recommendations'] = self.recommendations_calculator.get_recommendations(metrics)


        if mode == "dev":
            display_dev_mode(frame, metrics)
        elif mode == "user":
            display_user_mode(frame, metrics)

        return frame, metrics
    
    def calculate_body_angles(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], 
                              confidence_threshold: float, metrics: Dict[str, Any]):
        valid_keypoints = self.get_valid_keypoints(keypoint_coords, keypoint_confs, confidence_threshold)
        
        self.calculate_trunk_metrics(valid_keypoints, metrics)
        self.calculate_elbow_metrics(valid_keypoints, metrics)
        self.calculate_knee_metrics(valid_keypoints, metrics)
        self.calculate_arm_swing_metrics(valid_keypoints, metrics)
        self.calculate_hip_ankle_metrics(valid_keypoints, metrics)
    
    def get_valid_keypoints(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], 
                            confidence_threshold: float) -> Dict[int, np.ndarray]:
        return {i: coord for i, (coord, conf) in enumerate(zip(keypoint_coords, keypoint_confs)) 
                if conf > confidence_threshold}
    
    def calculate_trunk_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        if all(i in valid_keypoints for i in [0, 1, 2, 3]):
            shoulder_midpoint = (valid_keypoints[0] + valid_keypoints[1]) / 2
            hip_midpoint = (valid_keypoints[2] + valid_keypoints[3]) / 2
            metrics['trunk_angle'] = AngleCalculator.calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
            metrics['trunk_angle_assessment'] = Assessor.assess_torso_angle(abs(metrics['trunk_angle']))

            torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
            metrics['distance_cm'] = (KNOWN_TORSO_LENGTH_CM * FOCAL_LENGTH_PX) / torso_length_px
            
            self.hip_positions.append(hip_midpoint[1])
            metrics['vertical_oscillation'] = calculate_vertical_oscillation(
                list(self.hip_positions), metrics['distance_cm'] / FOCAL_LENGTH_PX)
            metrics['vertical_oscillation_assessment'] = Assessor.assess_vertical_oscillation(metrics['vertical_oscillation'])

            

    def calculate_elbow_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        # MoveNet keypoints: left_shoulder (5), left_elbow (7), left_wrist (9)
        if all(i in valid_keypoints for i in [5, 7, 9]):
            metrics['left_elbow_angle'] = AngleCalculator.calculate_elbow_angle(
                valid_keypoints[5], valid_keypoints[7], valid_keypoints[9]
            )
            metrics['left_elbow_angle_assessment'] = Assessor.assess_elbow_angle(metrics['left_elbow_angle'])

    def calculate_knee_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        # Left knee
        if all(i in valid_keypoints for i in [2, 4, 6]):
            metrics['left_knee_angle'] = AngleCalculator.calculate_knee_angle(
                valid_keypoints[2], valid_keypoints[4], valid_keypoints[6])
            self.last_left_knee_angle = metrics['left_knee_angle']
        # Right knee
        if all(i in valid_keypoints for i in [3, 5, 7]):
            metrics['right_knee_angle'] = AngleCalculator.calculate_knee_angle(
                valid_keypoints[3], valid_keypoints[5], valid_keypoints[7])
            self.last_right_knee_angle = metrics['right_knee_angle']
        
    def calculate_arm_swing_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        if all(i in valid_keypoints for i in [0, 8]):  # Assuming 0 is left shoulder and 8 is left elbow
            current_arm_swing_angle = AngleCalculator.calculate_arm_swing_angle(
                valid_keypoints[0], valid_keypoints[8])
            
            # Detect change in swing direction
            if self.last_arm_swing_direction is None:
                self.last_arm_swing_direction = current_arm_swing_angle > self.max_arm_swing_angle
            else:
                current_direction = current_arm_swing_angle > self.max_arm_swing_angle
                if current_direction != self.last_arm_swing_direction:
                    # Direction changed, assess the previous maximum
                    metrics['max_arm_swing_angle'] = self.max_arm_swing_angle
                    metrics['max_arm_swing_angle_assessment'] = Assessor.assess_arm_swing_angle(self.max_arm_swing_angle)
                    self.max_arm_swing_angle = current_arm_swing_angle
                self.last_arm_swing_direction = current_direction
            
            # Update maximum if current angle is greater
            if abs(current_arm_swing_angle - self.max_arm_swing_angle) > self.arm_swing_threshold:
                self.max_arm_swing_angle = max(self.max_arm_swing_angle, current_arm_swing_angle)
            
            metrics['arm_swing_angle'] = current_arm_swing_angle

    def calculate_arm_swing_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        if all(i in valid_keypoints for i in [0, 8]):
            metrics['arm_swing_angle'] = AngleCalculator.calculate_arm_swing_angle(
                valid_keypoints[0], valid_keypoints[8])
            metrics['arm_swing_angle_assessment'] = Assessor.assess_arm_swing_angle(metrics['arm_swing_angle'])


    def calculate_hip_ankle_metrics(self, valid_keypoints: Dict[int, np.ndarray], metrics: Dict[str, Any]):
        # Left hip-ankle angle
        if all(i in valid_keypoints for i in [2, 6]):
            metrics['left_hip_ankle_angle'] = AngleCalculator.calculate_hip_ankle_angle(
                valid_keypoints[2], valid_keypoints[6])
            self.last_left_hip_ankle_angle = metrics['left_hip_ankle_angle']
        # Right hip-ankle angle
        if all(i in valid_keypoints for i in [3, 7]):
            metrics['right_hip_ankle_angle'] = AngleCalculator.calculate_hip_ankle_angle(
                valid_keypoints[3], valid_keypoints[7])
            self.last_right_hip_ankle_angle = metrics['right_hip_ankle_angle']
    
    def detect_foot_strikes(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], 
                            confidence_threshold: float, timestamp: float, metrics: Dict[str, Any]):
        valid_keypoints = self.get_valid_keypoints(keypoint_coords, keypoint_confs, confidence_threshold)
        
        self.detect_left_foot_strike(valid_keypoints, timestamp, metrics)
        self.detect_right_foot_strike(valid_keypoints, timestamp, metrics)

    def detect_left_foot_strike(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, Any]):
        if 6 in valid_keypoints:  # Left ankle
            metrics['left_foot_strike'], _ = self.left_foot_strike_detector.update(valid_keypoints[6], timestamp)
            if metrics['left_foot_strike']:
                self.left_step_count += 1
                self.update_assessments(metrics, left_is_front=True)
            metrics['left_ankle_position'] = self.left_foot_strike_detector.get_filtered_position()
            metrics['left_ankle_position_unfiltered'] = valid_keypoints[6][0]

    def detect_right_foot_strike(self, valid_keypoints: Dict[int, np.ndarray], timestamp: float, metrics: Dict[str, Any]):
        if 7 in valid_keypoints:  # Right ankle
            metrics['right_foot_strike'], _ = self.right_foot_strike_detector.update(valid_keypoints[7], timestamp)
            if metrics['right_foot_strike']:
                self.right_step_count += 1
                self.update_assessments(metrics, left_is_front=False)
            metrics['right_ankle_position'] = self.right_foot_strike_detector.get_filtered_position()
            metrics['right_ankle_position_unfiltered'] = valid_keypoints[7][0]

    def update_assessments(self, metrics: Dict[str, Any], left_is_front: bool):
        if left_is_front:
            metrics['left_knee_assessment'] = Assessor.assess_knee_angle(self.last_left_knee_angle, is_front=True)
            metrics['right_knee_assessment'] = Assessor.assess_knee_angle(self.last_right_knee_angle, is_front=False)
            metrics['left_hip_ankle_angle_assessment'] = Assessor.assess_hip_ankle_angle(self.last_left_hip_ankle_angle)
        else:
            metrics['right_knee_assessment'] = Assessor.assess_knee_angle(self.last_right_knee_angle, is_front=True)
            metrics['left_knee_assessment'] = Assessor.assess_knee_angle(self.last_left_knee_angle, is_front=False)
            metrics['right_hip_ankle_angle_assessment'] = Assessor.assess_hip_ankle_angle(self.last_right_hip_ankle_angle)

    def calculate_step_metrics(self, metrics):
        self.total_step_count = self.left_step_count + self.right_step_count
        metrics['total_step_count'] = self.total_step_count
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60

    def get_available_metrics(self) -> Dict[str, Any]:
        """Return the dictionary of available metrics with their default values."""
        return self.available_metrics
    
def calculate_vertical_oscillation(hip_positions: List[float], scale_factor: float) -> float:
    """Calculate the vertical oscillation using a moving average."""
    if len(hip_positions) < 2:
        return 0.0
    moving_avg = np.convolve(hip_positions, np.ones(5), 'valid') / 5
    return (np.max(moving_avg) - np.min(moving_avg)) / 2 * scale_factor





    

