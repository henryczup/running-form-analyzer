import cv2
import numpy as np
import matplotlib.pyplot as plt

# Known parameters
KNOWN_TORSO_LENGTH_CM = 52  # Average height of a person in cm

# Camera parameters for your XPS 15 
HFOV_DEG = 74  # Horizontal field of view in degrees
IMAGE_WIDTH_PX = 1280  # Image width in pixels

# Convert HFOV to radians
HFOV_RAD = np.radians(HFOV_DEG)

# Calculate the focal length in pixels
FOCAL_LENGTH_PX = (IMAGE_WIDTH_PX / 2) / np.tan(HFOV_RAD / 2)

EDGES = {(0,1): 'm', 
         (0,2): 'c', 
         (1,3): 'm', 
         (2,4): 'c', 
         (0,5): 'm', 
         (0,6): 'c', 
         (5,7): 'm', 
         (7,9): 'm', 
         (6,8): 'c', 
         (8,10): 'c', 
         (5,6): 'y', 
         (5,11): 'm', 
         (6,12): 'c', 
         (11,12): 'y', 
         (11,13): 'm', 
         (13,15): 'm', 
         (12,14): 'c', 
         (14,16): 'c'
         }

def extract_keypoints(results, image_height, image_width):
    keypoints = results[0]
    left_shoulder_x = keypoints[5][1] * image_width
    left_shoulder_y = keypoints[5][0] * image_height
    left_shoulder_conf = keypoints[5][2]
    
    right_shoulder_x = keypoints[6][1] * image_width
    right_shoulder_y = keypoints[6][0] * image_height
    right_shoulder_conf = keypoints[6][2]
    
    left_hip_x = keypoints[11][1] * image_width
    left_hip_y = keypoints[11][0] * image_height
    left_hip_conf = keypoints[11][2]
    
    right_hip_x = keypoints[12][1] * image_width
    right_hip_y = keypoints[12][0] * image_height
    right_hip_conf = keypoints[12][2]
    
    left_knee_x = keypoints[13][1] * image_width
    left_knee_y = keypoints[13][0] * image_height
    left_knee_conf = keypoints[13][2]
    
    right_knee_x = keypoints[14][1] * image_width
    right_knee_y = keypoints[14][0] * image_height
    right_knee_conf = keypoints[14][2]
    
    left_ankle_x = keypoints[15][1] * image_width
    left_ankle_y = keypoints[15][0] * image_height
    left_ankle_conf = keypoints[15][2]
    
    right_ankle_x = keypoints[16][1] * image_width
    right_ankle_y = keypoints[16][0] * image_height
    right_ankle_conf = keypoints[16][2]
    
    left_elbow_x = keypoints[7][1] * image_width
    left_elbow_y = keypoints[7][0] * image_height
    left_elbow_conf = keypoints[7][2]
    
    right_elbow_x = keypoints[8][1] * image_width
    right_elbow_y = keypoints[8][0] * image_height
    right_elbow_conf = keypoints[8][2]

    # Return the keypoint coordinates and confidence scores separately
    left_shoulder = np.array([left_shoulder_x, left_shoulder_y])
    right_shoulder = np.array([right_shoulder_x, right_shoulder_y])
    left_hip = np.array([left_hip_x, left_hip_y])
    right_hip = np.array([right_hip_x, right_hip_y])
    left_knee = np.array([left_knee_x, left_knee_y])
    right_knee = np.array([right_knee_x, right_knee_y])
    left_ankle = np.array([left_ankle_x, left_ankle_y])
    right_ankle = np.array([right_ankle_x, right_ankle_y])
    left_elbow = np.array([left_elbow_x, left_elbow_y])
    right_elbow = np.array([right_elbow_x, right_elbow_y])
    
    keypoint_coords = [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_elbow, right_elbow]
    keypoint_confs = [left_shoulder_conf, right_shoulder_conf, left_hip_conf, right_hip_conf, left_knee_conf, right_knee_conf, left_ankle_conf, right_ankle_conf, left_elbow_conf, right_elbow_conf]
    
    return keypoint_coords, keypoint_confs



def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 

def draw_connections(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
    return frame


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

def calculate_metrics(keypoint_coords, keypoint_confs, frame, frame_count, confidence_threshold, hip_positions, left_foot_positions, right_foot_positions):
    left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_elbow, right_elbow = keypoint_coords
    left_shoulder_conf, right_shoulder_conf, left_hip_conf, right_hip_conf, left_knee_conf, right_knee_conf, left_ankle_conf, right_ankle_conf, left_elbow_conf, right_elbow_conf = keypoint_confs

    # Initialize distance_cm with a default value
    distance_cm = 0.0

    # Trunk angle calculation
    if left_shoulder_conf > confidence_threshold and right_shoulder_conf > confidence_threshold and left_hip_conf > confidence_threshold and right_hip_conf > confidence_threshold:
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        hip_midpoint = (left_hip + right_hip) / 2
        trunk_angle = calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
        cv2.putText(frame, f"Trunk Angle: {trunk_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Trunk Angle", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Knee angle calculation
    if left_hip_conf > confidence_threshold and left_knee_conf > confidence_threshold and left_ankle_conf > confidence_threshold:
        knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)
        cv2.putText(frame, f"Knee Angle: {knee_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Knee Angle", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Arm swing angle calculation
    if left_shoulder_conf > confidence_threshold and left_elbow_conf > confidence_threshold:
        arm_swing_angle = calculate_arm_swing_angle(left_shoulder, left_elbow)
        cv2.putText(frame, f"Arm Swing Angle: {arm_swing_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Arm Swing Angle", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Distance estimation
    if left_hip_conf > confidence_threshold and left_shoulder_conf > confidence_threshold and right_hip_conf > confidence_threshold and right_shoulder_conf > confidence_threshold:
        hip_midpoint = (left_hip + right_hip) / 2
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        torso_length_px = np.abs(hip_midpoint[1] - shoulder_midpoint[1])
        distance_cm = estimate_distance(torso_length_px, FOCAL_LENGTH_PX, KNOWN_TORSO_LENGTH_CM)
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Distance", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Vertical oscillation calculation
    if left_hip_conf > confidence_threshold and right_hip_conf > confidence_threshold:
        hip_midpoint = (left_hip + right_hip) / 2
        hip_positions.append(hip_midpoint[1])
        if len(hip_positions) > 10:
            hip_positions.pop(0)
        vertical_oscillation = calculate_vertical_oscillation(hip_positions, distance_cm / FOCAL_LENGTH_PX)
        cv2.putText(frame, f"Vertical Oscillation: {vertical_oscillation:.2f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Vertical Oscillation", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Hip to ankle angle calculation
    if left_hip_conf > confidence_threshold and left_ankle_conf > confidence_threshold:
        left_hip_ankle_angle = calculate_hip_ankle_angle(left_hip, left_ankle)
        cv2.putText(frame, f"Left Hip-Ankle Angle: {left_hip_ankle_angle:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Left Hip-Ankle Angle", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    if right_hip_conf > confidence_threshold and right_ankle_conf > confidence_threshold:
        right_hip_ankle_angle = calculate_hip_ankle_angle(right_hip, right_ankle)
        cv2.putText(frame, f"Right Hip-Ankle Angle: {right_hip_ankle_angle:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Cannot calculate Right Hip-Ankle Angle", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)


    return frame, hip_positions

def visualize_cadence(left_foot_positions, right_foot_positions, frame_count, threshold=5):
    if len(left_foot_positions) != len(right_foot_positions):
        raise ValueError("Left and right foot position arrays must have the same length.")

    time_data = np.arange(len(left_foot_positions)) / frame_count

    left_foot_velocities = calculate_velocities(left_foot_positions)
    left_gait_phases = detect_gait_phases(left_foot_velocities, threshold)

    plt.figure(figsize=(10, 6))
    # plt.plot(time_data[1:], left_foot_positions[1:], label='Left Foot')
    plt.plot(time_data, left_foot_positions, label='Left Foot')
    plt.plot(time_data, right_foot_positions, label='Right Foot')

    # Label gait phases and calculate durations for the left leg
    left_stance_duration = 0
    left_swing_duration = 0
    left_float_duration = 0

    for i in range(len(left_gait_phases)):
        if left_gait_phases[i] == 'Stance':
            plt.axvspan(time_data[i], time_data[i+1], color='green', alpha=0.2)
            left_stance_duration += time_data[i+1] - time_data[i]
        elif left_gait_phases[i] == 'Swing':
            plt.axvspan(time_data[i], time_data[i+1], color='blue', alpha=0.2)
            left_swing_duration += time_data[i+1] - time_data[i]
        elif left_gait_phases[i] == 'Float':
            plt.axvspan(time_data[i], time_data[i+1], color='red', alpha=0.2)
            left_float_duration += time_data[i+1] - time_data[i]

    plt.xlabel('Time')
    plt.ylabel('Foot Position')
    plt.title('Left and Right Foot Positions')
    plt.legend()

    plt.figtext(0.05, 0.95, f"Left Stance Duration: {left_stance_duration:.2f}s", fontsize=10)
    plt.figtext(0.05, 0.92, f"Left Swing Duration: {left_swing_duration:.2f}s", fontsize=10)
    plt.figtext(0.05, 0.89, f"Left Float Duration: {left_float_duration:.2f}s", fontsize=10)

    plt.tight_layout()
    plt.show()


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
