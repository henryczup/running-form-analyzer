import cv2

def display_dev_mode(frame, metrics):
    cv2.putText(frame, f"Trunk Angle: {metrics['trunk_angle']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Knee Angle: {metrics['knee_angle']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Arm Swing Angle: {metrics['arm_swing_angle']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Distance: {metrics['distance_cm']:.2f} cm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Vertical Oscillation: {metrics['vertical_oscillation']:.2f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Left Hip-Ankle Angle: {metrics['left_hip_ankle_angle']:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(frame, f"Right Hip-Ankle Angle: {metrics['right_hip_ankle_angle']:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

def display_user_mode(frame, metrics):
    if not (-15 <= metrics['trunk_angle'] <= 15):
        cv2.putText(frame, f"Trunk Angle: {metrics['trunk_angle']:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, "Recommendation: Keep your trunk stable", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if metrics['vertical_oscillation'] > 10:
        cv2.putText(frame, f"Vertical Oscillation: {metrics['vertical_oscillation']:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, "Recommendation: Decrease your vertical oscillation", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def display_gait_phases(frame, left_phase, right_phase, cadence):
    height, width, _ = frame.shape
    
    # Left leg gait phase
    cv2.putText(frame, f"Left Gait Phase: {left_phase.name if left_phase else 'N/A'}", 
                (10, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Right leg gait phase
    cv2.putText(frame, f"Right Gait Phase: {right_phase.name if right_phase else 'N/A'}", 
                (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Cadence
    cv2.putText(frame, f"Cadence: {cadence:.1f} steps/min", 
                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


import cv2
import numpy as np
from gait_phase import GaitPhase

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

# edges for the pose graph
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