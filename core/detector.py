import cv2
import numpy as np


def extract_keypoints(results, image_height, image_width):
    keypoints = results[0]
    
    # Extract keypoint coordinates and confidence scores
    keypoint_coords = []
    keypoint_confs = []
    
    for i in range(17):
        x = keypoints[i][1] * image_width
        y = keypoints[i][0] * image_height
        conf = keypoints[i][2]
        
        keypoint_coords.append(np.array([x, y]))
        keypoint_confs.append(conf)
    
    return keypoint_coords, keypoint_confs


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