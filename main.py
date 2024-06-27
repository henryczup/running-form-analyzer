import cv2
import numpy as np
import time
from typing import Tuple
import argparse

from config import THUNDER_PATH
from models.blazepose_model import BlazePoseModel
from models.lite_hrnet import LiteHRNetModel
from movenet import MoveNetModel
from detection import extract_keypoints
from measurements import MetricsCalculator
from display import draw_connections, draw_keypoints
from metric_logger import MetricsLogger
from video_recorder import VideoRecorder

class RunningAnalyzer:
    def __init__(self, model_type='movenet', model_path=THUNDER_PATH):
        if model_type == 'blazepose':
            self.model = BlazePoseModel()
        elif model_type == 'lite_hrnet':
            # TODO: fix this
            self.model = LiteHRNetModel()
        else:
            self.model = MoveNetModel(model_path)
        
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.frame_count = 0
        self.mode = "dev"
        self.metrics_calculator = MetricsCalculator()
        self.metrics_logger = MetricsLogger()
        self.video_recorder = VideoRecorder()

    def process_frame(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.frame_count += 1
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
                elif key == ord('t'):
                    self.metrics_calculator.set_filter_type("temporal")
                elif key == ord('k'):
                    self.metrics_calculator.set_filter_type("kalman")
                elif key == ord('n'):
                    self.metrics_calculator.set_filter_type("none")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.metrics_logger.close()
            self.video_recorder.stop_recording()
            print("\nPost-processing options:")
            self.video_recorder.post_recording_options()
            self.metrics_logger.post_logging_options()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Running Analysis using various pose estimation models")
    parser.add_argument('--model_type', type=str, default='movenet', choices=['movenet', 'blazepose', 'lite_hrnet'], help="Type of pose estimation model to use")
    parser.add_argument('--model_path', type=str, default=THUNDER_PATH, help="Path to the model (for MoveNet)")
    parser.add_argument('--filter_type', type=str, default='temporal', choices=['temporal', 'kalman', 'none'], help="Type of filter to use for heel strike detection")
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyzer = RunningAnalyzer(model_type=args.model_type, model_path=args.model_path, filter_type=args.filter_type)
    analyzer.run()

if __name__ == "__main__":
    main()