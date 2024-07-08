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

def get_valid_keypoints(keypoint_coords, keypoint_confs, confidence_threshold):
    """
    Filter keypoints based on a confidence threshold.
    
    :param keypoint_coords: List of keypoint coordinates
    :param keypoint_confs: List of keypoint confidence scores
    :param confidence_threshold: Minimum confidence score to consider a keypoint valid
    :return: Dictionary of valid keypoints with their indices as keys
    """
    return {i: coord for i, (coord, conf) in enumerate(zip(keypoint_coords, keypoint_confs)) 
            if conf > confidence_threshold}
