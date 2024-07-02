import cv2
import numpy as np
import time
from typing import Tuple
from io_utils.display import draw_connections, draw_keypoints
from io_utils.metric_logger import MetricsLogger
from metrics.calculator import Calculator
from models.movenet import MoveNetModel
from io_utils.video_recorder import VideoRecorder
from core.config import THUNDER_PATH
from models.blazepose_model import BlazePoseModel
from models.lite_hrnet import LiteHRNetModel
from core.detector import extract_keypoints


class Analyzer:
    def __init__(self, model_type='blazepose', model_path=THUNDER_PATH, filter_type='kalman', detection_axis='x'):
        if model_type == 'blazepose':
            self.model = BlazePoseModel()
        elif model_type == 'lite_hrnet':
            # TODO: fix this
            self.model = LiteHRNetModel()
        elif model_type == 'movenet':
            self.model = MoveNetModel(model_path)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.frame_count = 0
        self.mode = "dev"
        self.metrics_calculator = Calculator(filter_type=filter_type, detection_axis=detection_axis)
        self.metrics_logger = MetricsLogger()
        self.video_recorder = VideoRecorder()

        # Initialize metrics logging with all available metrics
        available_metrics = self.metrics_calculator.get_available_metrics()
        self.metrics_logger.initialize_logging(available_metrics)

    def process_frame(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        current_time = time.time() - self.start_time

        # Start recording if it hasn't started yet
        if not self.video_recorder.recording:
            self.video_recorder.start_recording(frame)

        # Process frame
        keypoints_with_scores = self.model.predict(frame)
        
        if keypoints_with_scores is not None:
            keypoint_coords, keypoint_confs = extract_keypoints(keypoints_with_scores, frame.shape[0], frame.shape[1])
            draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.3)
            draw_connections(frame, keypoints_with_scores, confidence_threshold=0.3)
        else:
            # If no keypoints detected, use empty lists for coords and confs
            keypoint_coords, keypoint_confs = [], []

        # Calculate metrics (this will return zero metrics if keypoints are empty)
        frame, metrics = self.metrics_calculator.calculate_metrics(
            keypoint_coords, keypoint_confs, frame, 0.4, current_time, self.mode
        )

        # Log metrics
        self.metrics_logger.log_metrics(current_time, metrics)

        # Record the processed frame
        self.video_recorder.record_frame(frame)

        return True, frame

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.process_frame()
                if not ret:
                    break

                cv2.imshow("Running Analysis", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.mode = "dev"
                elif key == ord('u'):
                    self.mode = "user"
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.video_recorder.stop_recording()
            
            print("\nPost-processing options:")
            self.video_recorder.post_recording_options()
            self.metrics_logger.close()
            self.metrics_logger.post_logging_options()