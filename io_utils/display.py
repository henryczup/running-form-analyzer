import cv2
import numpy as np

from core.config import EDGES

def display_dev_mode(frame, metrics):
    cv2.putText(frame, f"Trunk Angle: {metrics['trunk_angle']:.2f} ({metrics['trunk_angle_assessment']})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Elbow Angle: {metrics['left_elbow_angle']:.2f} ({metrics['left_elbow_angle_assessment']})", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)    
    cv2.putText(frame, f"Left Knee Angle: {metrics['left_knee_angle']:.2f} ({metrics['left_knee_assessment']})", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Right Knee Angle: {metrics['right_knee_angle']:.2f} ({metrics['right_knee_assessment']})", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Arm Swing Angle: {metrics['max_left_arm_swing_angle']:.2f} ({metrics['max_left_arm_swing_angle_assessment']})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Distance: {metrics['distance_cm']:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Vertical Oscillation: {metrics['vertical_oscillation']:.2f} cm ({metrics['vertical_oscillation_assessment']})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Left Hip-Ankle Angle: {metrics['left_hip_ankle_angle']:.2f} ({metrics['left_hip_ankle_angle_assessment']})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Right Hip-Ankle Angle: {metrics['right_hip_ankle_angle']:.2f} ({metrics['right_hip_ankle_angle_assessment']})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Total Step Count: {metrics['total_step_count']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Steps per Minute: {metrics['steps_per_minute']:.2f}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Elapsed Time: {metrics['elapsed_time']:.2f} s", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Left Foot Strike: {metrics['left_foot_strike']}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Right Foot Strike: {metrics['right_foot_strike']}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def display_user_mode(frame, metrics):
    y_offset = 30
    line_spacing = 40

    recommendations = metrics['recommendations']

    for recommendation in recommendations:
        cv2.putText(frame, recommendation, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        y_offset += line_spacing

    if not recommendations:
        cv2.putText(frame, "Great form! Keep it up!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


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