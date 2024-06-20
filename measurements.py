import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import HFOV_DEG, IMAGE_WIDTH_PX, KNOWN_TORSO_LENGTH_CM
from visualization import display_dev_mode, display_user_mode

# Convert HFOV to radians
HFOV_RAD = np.radians(HFOV_DEG)

# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)

def calculate_trunk_angle(shoulder, hip):
    trunk_line = hip - shoulder
    vertical_line = np.array([0, 1])
    
    dot_product = np.dot(trunk_line, vertical_line)
    magnitudes = np.linalg.norm(trunk_line) * np.linalg.norm(vertical_line)
    
    angle_rad = np.arccos(dot_product / magnitudes)
    angle_deg = np.degrees(angle_rad)
    
    # Determine the orientation of the trunk line
    if trunk_line[0] < 0:
        angle_deg = -angle_deg
    
    return angle_deg

def calculate_knee_angle(hip, knee, ankle):
    thigh = np.array(hip) - np.array(knee)
    lower_leg = np.array(knee) - np.array(ankle)
    
    dot_product = np.dot(thigh, lower_leg)
    magnitudes = np.linalg.norm(thigh) * np.linalg.norm(lower_leg)
    
    angle_rad = np.arccos(dot_product / magnitudes)
    angle_deg = np.degrees(angle_rad)
    
    return 180 - angle_deg

def calculate_arm_swing_angle(shoulder, elbow):
    upper_arm = np.array(elbow) - np.array(shoulder)
    vertical_line = np.array([0, 1])
    
    dot_product = np.dot(upper_arm, vertical_line)
    magnitudes = np.linalg.norm(upper_arm) * np.linalg.norm(vertical_line)
    
    angle_rad = np.arccos(dot_product / magnitudes)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_hip_ankle_angle(hip, ankle):
    hip_ankle_line = np.array(ankle) - np.array(hip)
    vertical_line = np.array([0, 1])
    
    dot_product = np.dot(hip_ankle_line, vertical_line)
    magnitudes = np.linalg.norm(hip_ankle_line) * np.linalg.norm(vertical_line)
    
    angle_rad = np.arccos(dot_product / magnitudes)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_vertical_oscillation(hip_positions, scale_factor):
    max_position = np.max(hip_positions)
    min_position = np.min(hip_positions)
    vertical_oscillation = (max_position - min_position) / 2
    return vertical_oscillation * scale_factor  # Convert to centimeters

def estimate_distance(height_px, focal_length_px, known_height_cm):
    # Use the known height and focal length to estimate the distance
    distance_cm = (known_height_cm * focal_length_px) / height_px
    return distance_cm


def calculate_metrics(keypoint_coords, keypoint_confs, frame, confidence_threshold, hip_positions, left_ankle_positions, right_ankle_positions, mode, left_ankle_velocities):
    left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_elbow, right_elbow = keypoint_coords
    left_shoulder_conf, right_shoulder_conf, left_hip_conf, right_hip_conf, left_knee_conf, right_knee_conf, left_ankle_conf, right_ankle_conf, left_elbow_conf, right_elbow_conf = keypoint_confs

    trunk_angle = 0.0
    knee_angle = 0.0
    arm_swing_angle = 0.0
    distance_cm = 0.0
    vertical_oscillation = 0.0
    left_hip_ankle_angle = 0.0
    right_hip_ankle_angle = 0.0

    # Trunk angle calculation
    if left_shoulder_conf > confidence_threshold and right_shoulder_conf > confidence_threshold and left_hip_conf > confidence_threshold and right_hip_conf > confidence_threshold:
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2
        trunk_angle = calculate_trunk_angle(shoulder_midpoint, hip_midpoint)

    # Knee angle calculation
    if left_hip_conf > confidence_threshold and left_knee_conf > confidence_threshold and left_ankle_conf > confidence_threshold:
        knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)

    # Arm swing angle calculation
    if left_shoulder_conf > confidence_threshold and left_elbow_conf > confidence_threshold:
        arm_swing_angle = calculate_arm_swing_angle(left_shoulder, left_elbow)

    # Distance estimation
    if left_hip_conf > confidence_threshold and left_shoulder_conf > confidence_threshold and right_hip_conf > confidence_threshold and right_shoulder_conf > confidence_threshold:
        hip_midpoint = (left_hip + right_hip) / 2
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
        distance_cm = estimate_distance(torso_length_px, FOCAL_LENGTH_PX, KNOWN_TORSO_LENGTH_CM)

    # Vertical oscillation calculation
    if left_hip_conf > confidence_threshold and right_hip_conf > confidence_threshold:
        hip_midpoint = (left_hip + right_hip) / 2
        hip_positions.append(hip_midpoint[1])
        if len(hip_positions) > 10:
            hip_positions.pop(0)
        vertical_oscillation = calculate_vertical_oscillation(hip_positions, distance_cm / FOCAL_LENGTH_PX)

    # Hip to ankle angle calculation
    if left_hip_conf > confidence_threshold and left_ankle_conf > confidence_threshold:
        left_hip_ankle_angle = calculate_hip_ankle_angle(left_hip, left_ankle)

    if right_hip_conf > confidence_threshold and right_ankle_conf > confidence_threshold:
        right_hip_ankle_angle = calculate_hip_ankle_angle(right_hip, right_ankle)

    if mode == "dev":
        display_dev_mode(frame, trunk_angle, knee_angle, arm_swing_angle, distance_cm, vertical_oscillation, left_hip_ankle_angle, right_hip_ankle_angle)
    elif mode == "user":
        display_user_mode(frame, trunk_angle, vertical_oscillation)

    return frame, hip_positions, left_ankle_positions, left_ankle_velocities


def calculate_velocities(positions):
    velocities = []
    for i in range(1, len(positions)):
        velocity = positions[i] - positions[i-1]
        velocities.append(velocity)
    return velocities

def detect_gait_phases(foot_velocities, threshold):
    gait_phases = []
    for velocity in foot_velocities:
        if abs(velocity) < threshold:
            gait_phases.append('Stance')
        elif velocity >= threshold:
            gait_phases.append('Swing')
        else:
            gait_phases.append('Float')
    return gait_phases
