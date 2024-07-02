from typing import List, Dict
import numpy as np

def get_valid_keypoints(keypoint_coords: List[np.ndarray], keypoint_confs: List[float], 
                        confidence_threshold: float) -> Dict[int, np.ndarray]:
    return {i: coord for i, (coord, conf) in enumerate(zip(keypoint_coords, keypoint_confs)) 
            if conf > confidence_threshold}