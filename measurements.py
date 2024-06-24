import cv2
import numpy as np
from typing import List, Tuple, Dict
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

class GaitCycleDetector:
    def __init__(self, window_size: int = 30):
        self.ankle_positions = deque(maxlen=window_size)
        self.knee_positions = deque(maxlen=window_size)
        self.hip_positions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.current_phase = GaitPhase.MID_STANCE
        self.last_heel_strike_time = None
        self.stride_times = deque(maxlen=5)

    def update(self, ankle_position: float, knee_position: float, hip_position: float, timestamp: float) -> Tuple[GaitPhase, float]:
        self.ankle_positions.append(ankle_position)
        self.knee_positions.append(knee_position)
        self.hip_positions.append(hip_position)
        self.timestamps.append(timestamp)

        cadence = 0.0

        if len(self.ankle_positions) < 3:
            return self.current_phase, cadence

        # Detect heel strike (Initial Contact)
        if (self.ankle_positions[-2] < self.ankle_positions[-1] and 
            self.ankle_positions[-2] < self.ankle_positions[-3]):
            if self.last_heel_strike_time is None or (timestamp - self.last_heel_strike_time) > 0.4:  # Prevent false positives
                if self.last_heel_strike_time is not None:
                    stride_time = timestamp - self.last_heel_strike_time
                    self.stride_times.append(stride_time)
                self.current_phase = GaitPhase.INITIAL_CONTACT
                self.last_heel_strike_time = timestamp

        # Detect other phases
        elif self.current_phase == GaitPhase.INITIAL_CONTACT:
            if self.knee_positions[-1] > self.knee_positions[-2]:
                self.current_phase = GaitPhase.LOADING_RESPONSE
        elif self.current_phase == GaitPhase.LOADING_RESPONSE:
            if self.ankle_positions[-1] < self.ankle_positions[-2]:
                self.current_phase = GaitPhase.MID_STANCE
        elif self.current_phase == GaitPhase.MID_STANCE:
            if self.ankle_positions[-1] < self.ankle_positions[-2]:
                self.current_phase = GaitPhase.TERMINAL_STANCE
        elif self.current_phase == GaitPhase.TERMINAL_STANCE:
            if self.ankle_positions[-1] > self.ankle_positions[-2]:
                self.current_phase = GaitPhase.PRE_SWING
        elif self.current_phase == GaitPhase.PRE_SWING:
            if self.knee_positions[-1] > self.knee_positions[-2]:
                self.current_phase = GaitPhase.INITIAL_SWING
        elif self.current_phase == GaitPhase.INITIAL_SWING:
            if self.knee_positions[-1] < self.knee_positions[-2]:
                self.current_phase = GaitPhase.MID_SWING
        elif self.current_phase == GaitPhase.MID_SWING:
            if self.ankle_positions[-1] < self.ankle_positions[-2]:
                self.current_phase = GaitPhase.TERMINAL_SWING

        # Calculate cadence
        if len(self.stride_times) > 0:
            average_stride_time = sum(self.stride_times) / len(self.stride_times)
            cadence = 60 / average_stride_time  # steps per minute

        return self.current_phase, cadence
    
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
    def __init__(self, hip_position_window=10):
        self.hip_positions = deque(maxlen=hip_position_window)
        self.fps_tracker = FPSTracker()
        self.left_gait_detector = GaitCycleDetector()
        self.right_gait_detector = GaitCycleDetector()

    def calculate_metrics(self, keypoint_coords: List[np.ndarray], keypoint_confs: List[float], frame: np.ndarray, 
                          confidence_threshold: float, timestamp: float, mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Calculate various metrics based on keypoint data."""
        metrics = {
            'trunk_angle': 0.0,
            'knee_angle': 0.0,
            'arm_swing_angle': 0.0,
            'distance_cm': 0.0,
            'vertical_oscillation': 0.0,
            'left_hip_ankle_angle': 0.0,
            'right_hip_ankle_angle': 0.0,
            'left_gait_phase': None,
            'right_gait_phase': None,
            'cadence': 0.0,
            'fps': 0.0
        }

        # Update FPS
        self.fps_tracker.update(timestamp)
        metrics['fps'] = self.fps_tracker.get_fps()

        if all(conf > confidence_threshold for conf in keypoint_confs[:4]):  # shoulders and hips
            shoulder_midpoint = (keypoint_coords[0] + keypoint_coords[1]) / 2
            hip_midpoint = (keypoint_coords[2] + keypoint_coords[3]) / 2
            metrics['trunk_angle'] = calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
            
            torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
            metrics['distance_cm'] = estimate_distance(torso_length_px, FOCAL_LENGTH_PX, KNOWN_TORSO_LENGTH_CM)
            
            self.hip_positions.append(hip_midpoint[1])
            metrics['vertical_oscillation'] = calculate_vertical_oscillation(list(self.hip_positions), metrics['distance_cm'] / FOCAL_LENGTH_PX)

        if all(conf > confidence_threshold for conf in [keypoint_confs[2], keypoint_confs[4], keypoint_confs[6]]):  # left hip, knee, ankle
            metrics['knee_angle'] = calculate_knee_angle(keypoint_coords[2], keypoint_coords[4], keypoint_coords[6])

        if all(conf > confidence_threshold for conf in [keypoint_confs[0], keypoint_confs[8]]):  # left shoulder, elbow
            metrics['arm_swing_angle'] = calculate_arm_swing_angle(keypoint_coords[0], keypoint_coords[8])

        if keypoint_confs[2] > confidence_threshold and keypoint_confs[6] > confidence_threshold:  # left hip, ankle
            metrics['left_hip_ankle_angle'] = calculate_hip_ankle_angle(keypoint_coords[2], keypoint_coords[6])

        if keypoint_confs[3] > confidence_threshold and keypoint_confs[7] > confidence_threshold:  # right hip, ankle
            metrics['right_hip_ankle_angle'] = calculate_hip_ankle_angle(keypoint_coords[3], keypoint_coords[7])

        # Calculate gait cycle phases and cadence
        left_cadence = right_cadence = 0
        if all(conf > confidence_threshold for conf in [keypoint_confs[2], keypoint_confs[4], keypoint_confs[6]]):  # left hip, knee, ankle
            metrics['left_gait_phase'], left_cadence = self.left_gait_detector.update(
                keypoint_coords[6][1],  # left ankle
                keypoint_coords[4][1],  # left knee
                keypoint_coords[2][1],  # left hip
                timestamp
            )

        if all(conf > confidence_threshold for conf in [keypoint_confs[3], keypoint_confs[5], keypoint_confs[7]]):  # right hip, knee, ankle
            metrics['right_gait_phase'], right_cadence = self.right_gait_detector.update(
                keypoint_coords[7][1],  # right ankle
                keypoint_coords[5][1],  # right knee
                keypoint_coords[3][1],  # right hip
                timestamp
            )

        # Average cadence from both legs
        metrics['cadence'] = (left_cadence + right_cadence) / 2 if left_cadence and right_cadence else max(left_cadence, right_cadence)

        if mode == "dev":
            display_dev_mode(frame, metrics)
        elif mode == "user":
            display_user_mode(frame, metrics)

        return frame, metrics



