import cv2
import numpy as np
import time
from typing import Tuple
from metrics.metrics import Metrics
from visualization.display import display_mode, display_recommendations, display_show_key_metrics
from visualization.metric_logger import MetricsLogger
from models.movenet import MoveNetModel
from visualization.video_recorder import VideoRecorder
from core.config import THUNDER_PATH, Config
from models.blazepose_model import BlazePoseModel
from models.lite_hrnet import LiteHRNetModel
from core.detector import extract_keypoints, get_valid_keypoints
from visualization.pose_drawer import draw_connections, draw_keypoints


class Analyzer:
    def __init__(self, config: Config):
        self.config = config
        model_type = config.model_type
        self.side = config.side
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.frame_count = 0
        self.mode = "metrics"
        self.metrics_calculator = Metrics(config)
        self.metrics_logger = MetricsLogger()
        self.video_recorder = VideoRecorder()

        if model_type == 'blazepose':
            self.model = BlazePoseModel()
        elif model_type == 'lite_hrnet':
            # TODO: fix this
            self.model = LiteHRNetModel()
        elif model_type == 'movenet':
            self.model = MoveNetModel(THUNDER_PATH)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        # Initialize metrics logging with all available metrics
        available_metrics = self.metrics_calculator.get_key_metrics()
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
        model_output = self.model.predict(frame)
        
        if model_output is not None:
            # Draw pose on frame
            draw_keypoints(frame, model_output, confidence_threshold=0.3)
            draw_connections(frame, model_output, confidence_threshold=0.3)

            # Extract and filter valid keypoints 
            # TODO: add filter for keypoints
            keypoint_coords, keypoint_confs = extract_keypoints(model_output, frame.shape[0], frame.shape[1])
            valid_keypoints = get_valid_keypoints(keypoint_coords, keypoint_confs, confidence_threshold=0.3)
        else:
           # If no keypoints are detected, set valid_keypoints to an empty dictionary
           valid_keypoints = {}

        # Calculate metrics (this will return zero metrics if keypoints are empty)
        metrics = self.metrics_calculator.calculate_metrics(valid_keypoints, current_time)

        # Log metrics
        self.metrics_logger.log_metrics(current_time, metrics)

        # Display metrics or recommendations based on the current mode
        frame = display_mode(frame, metrics, self.mode, self.side)

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
                elif key == ord('a'):
                    self.display_mode = "angles"
                elif key == ord('m'):
                    self.display_mode = "metrics"
                elif key == ord('r'):
                    self.display_mode = "recommendations"

                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.video_recorder.stop_recording()
            
            print("\nPost-processing options:")
            self.video_recorder.post_recording_options()
            self.metrics_logger.close()
            self.metrics_logger.post_logging_options()