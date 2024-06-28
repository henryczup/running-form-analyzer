import cv2
import numpy as np
from typing import Any, List, Literal, Tuple, Dict
from collections import deque
from gait_phase import GaitPhase
from config import HFOV_DEG, IMAGE_WIDTH_PX, KNOWN_TORSO_LENGTH_CM
from display import display_dev_mode, display_user_mode
import time

# Convert HFOV to radians
HFOV_RAD = np.radians(HFOV_DEG)

# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)

class FPSTracker:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
    
    def update(self, current_time):
        self.frame_times.append(current_time)
    
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

fps_tracker = FPSTracker()

class KalmanFilter:
    def __init__(self, initial_state, initial_estimate_error, measurement_noise, process_noise):
        self.state = initial_state
        self.estimate_error = initial_estimate_error
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise

    def update(self, measurement):
        # Prediction
        predicted_state = self.state
        predicted_estimate_error = self.estimate_error + self.process_noise

        # Update
        kalman_gain = predicted_estimate_error / (predicted_estimate_error + self.measurement_noise)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.estimate_error = (1 - kalman_gain) * predicted_estimate_error

        return self.state
    
class FootStrikeDetector:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "kalman", 
                 window_size: int = 10, smoothing_window: int = 5,
                 detection_axis: Literal["x", "y"] = "x"):
        self.filter_type = filter_type
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.detection_axis = detection_axis
        self.positions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.last_foot_strike_time = None
        
        if self.filter_type == "kalman":
            self.kalman = KalmanFilter(
                initial_state=0,
                initial_estimate_error=1,
                measurement_noise=0.1,
                process_noise=0.01
            )

    def temporal_filter(self, new_position: float) -> float:
        """Apply a simple moving average filter to the new position."""
        self.positions.append(new_position)
        if len(self.positions) < self.smoothing_window:
            return new_position
        return np.mean(list(self.positions)[-self.smoothing_window:])

    def update(self, ankle_position: np.ndarray, timestamp: float) -> Tuple[bool, float]:
        position = ankle_position[0] if self.detection_axis == "x" else ankle_position[1]
        
        if self.filter_type == "temporal":
            filtered_position = self.temporal_filter(position)
        elif self.filter_type == "kalman":
            filtered_position = self.kalman.update(position)
        else:  # no filter
            filtered_position = position
        
        self.positions.append(filtered_position)
        self.timestamps.append(timestamp)

        foot_strike_detected = False
        stride_time = 0.0

        if len(self.positions) < 3:
            return foot_strike_detected, stride_time

        # Detect foot strike based on the chosen axis
        if self.detection_axis == "x":
            # Foot strike when ankle x-position is at its minimum
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3]):
                foot_strike_detected = True
        else:  # "y" axis
            # Foot strike when ankle y-position starts increasing after decreasing
            if (self.positions[-2] < self.positions[-1] and 
                self.positions[-2] < self.positions[-3]):
                foot_strike_detected = True

        if foot_strike_detected:
            if self.last_foot_strike_time is None or (timestamp - self.last_foot_strike_time) > 0.4:  # Prevent false positives
                if self.last_foot_strike_time is not None:
                    stride_time = timestamp - self.last_foot_strike_time
                self.last_foot_strike_time = timestamp
            else:
                foot_strike_detected = False  # Reset if it's too soon after the last detection

        return foot_strike_detected, stride_time

    def get_filtered_position(self) -> float:
        return self.positions[-1] if self.positions else None

    def set_filter_type(self, filter_type: Literal["temporal", "kalman", "none"]):
        if filter_type not in ["temporal", "kalman", "none"]:
            raise ValueError("Filter type must be either 'temporal', 'kalman', or 'none'")
        self.filter_type = filter_type
        if filter_type == "kalman":
            self.kalman = KalmanFilter(
                initial_state=self.get_filtered_position() or 0,
                initial_estimate_error=1,
                measurement_noise=0.1,
                process_noise=0.01
            )

    def set_detection_axis(self, detection_axis: Literal["x", "y"]):
        if detection_axis not in ["x", "y"]:
            raise ValueError("Detection axis must be either 'x' or 'y'")
        self.detection_axis = detection_axis

    
def calculate_angle(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate the angle between two vectors."""
    dot_product = np.dot(vector1, vector2)
    magnitudes = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / magnitudes)
    return np.degrees(angle_rad)

def calculate_trunk_angle(shoulder: np.ndarray, hip: np.ndarray) -> float:
    """Calculate the trunk angle relative to vertical."""
    trunk_line = hip - shoulder
    vertical_line = np.array([0, 1])
    angle = calculate_angle(trunk_line, vertical_line)
    return -angle if trunk_line[0] < 0 else angle

def calculate_knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """Calculate the knee angle."""
    thigh = hip - knee
    lower_leg = ankle - knee
    return 180 - calculate_angle(thigh, lower_leg)

def calculate_arm_swing_angle(shoulder: np.ndarray, elbow: np.ndarray) -> float:
    """Calculate the arm swing angle relative to vertical."""
    upper_arm = elbow - shoulder
    vertical_line = np.array([0, 1])
    return calculate_angle(upper_arm, vertical_line)

def calculate_hip_ankle_angle(hip: np.ndarray, ankle: np.ndarray) -> float:
    """Calculate the hip to ankle angle relative to vertical."""
    hip_ankle_line = ankle - hip
    vertical_line = np.array([0, 1])
    return calculate_angle(hip_ankle_line, vertical_line)

def calculate_vertical_oscillation(hip_positions: List[float], scale_factor: float) -> float:
    """Calculate the vertical oscillation using a moving average."""
    if len(hip_positions) < 2:
        return 0.0
    moving_avg = np.convolve(hip_positions, np.ones(5), 'valid') / 5
    return (np.max(moving_avg) - np.min(moving_avg)) / 2 * scale_factor

def estimate_distance(height_px: float, focal_length_px: float, known_height_cm: float) -> float:
    """Estimate the distance based on known height and focal length."""
    return (known_height_cm * focal_length_px) / height_px

class MetricsCalculator:
    def __init__(self, filter_type: Literal["temporal", "kalman", "none"] = "temporal",
                 detection_axis: Literal["x", "y"] = "y"):
        self.hip_positions = deque(maxlen=10)
        self.fps_tracker = FPSTracker()
        self.left_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.right_foot_strike_detector = FootStrikeDetector(filter_type=filter_type, detection_axis=detection_axis)
        self.left_step_count = 0
        self.right_step_count = 0
        self.total_step_count = 0
        self.start_time = None
        self.available_metrics = {
            'trunk_angle': 0.0,
            'knee_angle': 0.0,
            'arm_swing_angle': 0.0,
            'distance_cm': 0.0,
            'vertical_oscillation': 0.0,
            'left_hip_ankle_angle': 0.0,
            'right_hip_ankle_angle': 0.0,
            'left_foot_strike': False,
            'right_foot_strike': False,
            'left_ankle_position': 0.0,
            'right_ankle_position': 0.0,
            'left_ankle_position_unfiltered': 0.0,
            'right_ankle_position_unfiltered': 0.0,
            'total_step_count': 0,
            'steps_per_minute': 0.0,
            'fps': 0.0,
            'elapsed_time': 0.0
        }

    def calculate_metrics(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], frame: np.ndarray, 
                          confidence_threshold: float, timestamp: float, mode: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calculate various metrics based on keypoint data."""
        if self.start_time is None:
            self.start_time = timestamp

        metrics = {metric: default_value for metric, default_value in self.available_metrics.items()}

        # Update FPS
        self.fps_tracker.update(timestamp)
        metrics['fps'] = self.fps_tracker.get_fps()

        # Calculate elapsed time
        metrics['elapsed_time'] = timestamp - self.start_time

        def is_keypoint_valid(index):
            return index < len(keypoint_coords) and index < len(keypoint_confs) and keypoint_confs[index] > confidence_threshold

        # Calculate trunk angle and distance
        if all(is_keypoint_valid(i) for i in [0, 1, 2, 3]):  # shoulders and hips
            shoulder_midpoint = (keypoint_coords[0] + keypoint_coords[1]) / 2
            hip_midpoint = (keypoint_coords[2] + keypoint_coords[3]) / 2
            metrics['trunk_angle'] = calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
            
            torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
            metrics['distance_cm'] = estimate_distance(torso_length_px, FOCAL_LENGTH_PX, KNOWN_TORSO_LENGTH_CM)
            
            self.hip_positions.append(hip_midpoint[1])
            metrics['vertical_oscillation'] = calculate_vertical_oscillation(list(self.hip_positions), metrics['distance_cm'] / FOCAL_LENGTH_PX)

        # Calculate knee angle (using left leg as an example)
        if all(is_keypoint_valid(i) for i in [2, 4, 6]):  # left hip, knee, ankle
            metrics['knee_angle'] = calculate_knee_angle(keypoint_coords[2], keypoint_coords[4], keypoint_coords[6])

        # Calculate arm swing angle (using left arm as an example)
        if all(is_keypoint_valid(i) for i in [0, 8]):  # left shoulder, elbow
            metrics['arm_swing_angle'] = calculate_arm_swing_angle(keypoint_coords[0], keypoint_coords[8])

        # Calculate hip-ankle angles
        if is_keypoint_valid(2) and is_keypoint_valid(6):  # left hip, ankle
            metrics['left_hip_ankle_angle'] = calculate_hip_ankle_angle(keypoint_coords[2], keypoint_coords[6])
        if is_keypoint_valid(3) and is_keypoint_valid(7):  # right hip, ankle
            metrics['right_hip_ankle_angle'] = calculate_hip_ankle_angle(keypoint_coords[3], keypoint_coords[7])

        # track left and right ankle positions filtered and unfiltered
        if is_keypoint_valid(6):  # left ankle
            metrics['left_ankle_position_unfiltered'] = keypoint_coords[6][0]
            metrics['left_ankle_position'] = self.left_foot_strike_detector.get_filtered_position()
        if is_keypoint_valid(7):  # right ankle
            metrics['right_ankle_position_unfiltered'] = keypoint_coords[7][0]
            metrics['right_ankle_position'] = self.right_foot_strike_detector.get_filtered_position()
        
        # Detect foot strikes and count steps
        if is_keypoint_valid(6):  # left ankle
            metrics['left_foot_strike'], _ = self.left_foot_strike_detector.update(keypoint_coords[6], timestamp)
            if metrics['left_foot_strike']:
                self.left_step_count += 1
            metrics['left_ankle_position'] = self.left_foot_strike_detector.get_filtered_position()

        if is_keypoint_valid(7):  # right ankle
            metrics['right_foot_strike'], _ = self.right_foot_strike_detector.update(keypoint_coords[7], timestamp)
            if metrics['right_foot_strike']:
                self.right_step_count += 1
            metrics['right_ankle_position'] = self.right_foot_strike_detector.get_filtered_position()

        # Calculate total step count
        self.total_step_count = self.left_step_count + self.right_step_count
        metrics['total_step_count'] = self.total_step_count

        # Calculate steps per minute
        if metrics['elapsed_time'] > 0:
            metrics['steps_per_minute'] = (self.total_step_count / metrics['elapsed_time']) * 60

        if mode == "dev":
            display_dev_mode(frame, metrics)
        elif mode == "user":
            display_user_mode(frame, metrics)

        return frame, metrics
    
    def set_filter_type(self, filter_type: Literal["temporal", "kalman", "none"]):
        self.left_foot_strike_detector.set_filter_type(filter_type)
        self.right_foot_strike_detector.set_filter_type(filter_type)

    def set_detection_axis(self, detection_axis: Literal["x", "y"]):
        self.left_foot_strike_detector.set_detection_axis(detection_axis)
        self.right_foot_strike_detector.set_detection_axis(detection_axis)

    def get_available_metrics(self) -> Dict[str, Any]:
        """Return the dictionary of available metrics with their default values."""
        return self.available_metrics



    

