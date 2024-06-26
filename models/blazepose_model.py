import mediapipe as mp
import cv2
import numpy as np

class BlazePoseModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return self.convert_blazepose_to_movenet_format(results.pose_landmarks, image.shape)
        return None

    def draw_pose(self, image, pose_landmarks):
        if pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                image,
                self.mp_pose.PoseLandmark(pose_landmarks),
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        return image

    @staticmethod
    def convert_blazepose_to_movenet_format(landmarks, image_shape):
        height, width, _ = image_shape
        movenet_keypoints = np.zeros((17, 3))

        # Updated BlazePose to MoveNet keypoint mapping
        blazepose_to_movenet = {
            0: 0,   # nose
            2: 1,   # left_eye
            5: 2,   # right_eye
            7: 3,   # left_ear
            8: 4,   # right_ear
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16, # right_ankle
        }

        for blazepose_idx, movenet_idx in blazepose_to_movenet.items():
            landmark = landmarks.landmark[blazepose_idx]
            movenet_keypoints[movenet_idx] = [
                landmark.y,
                landmark.x,
                landmark.visibility
            ]

        return movenet_keypoints.reshape(1, 17, 3)

    @staticmethod
    def convert_blazepose_to_keypoints(landmarks):
        if landmarks is not None:
            keypoint_coords = [(lm[1], lm[0]) for lm in landmarks[0]]
            keypoint_confs = [lm[2] for lm in landmarks[0]]
            return keypoint_coords, keypoint_confs
        return None, None