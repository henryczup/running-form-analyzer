import os
import cv2
import numpy as np
import time
from typing import List, Tuple
import argparse

from config import THUNDER_PATH
from movenet import MoveNetModel
from detection import extract_keypoints
from measurements import calculate_metrics, GaitCycleDetector
from display import display_gait_phases, draw_connections, draw_keypoints
from metric_logger import MetricsLogger
from video_recorder import VideoRecorder

class RunningAnalyzer:
    def __init__(self, model_path: str):
        self.model = MoveNetModel(model_path)
        self.cap = cv2.VideoCapture(0)
        self.hip_positions: List[float] = []
        self.start_time = time.time()
        self.frame_count = 0
        self.mode = "dev"
        self.left_gait_detector = GaitCycleDetector()
        self.right_gait_detector = GaitCycleDetector()
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

        # Make predictions
        keypoints_with_scores = self.model.predict(frame)

        # Draw the keypoints and connections on the frame
        draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.3)
        draw_connections(frame, keypoints_with_scores, confidence_threshold=0.3)

        # Extract keypoint coordinates and confidence scores
        keypoint_coords, keypoint_confs = extract_keypoints(keypoints_with_scores, frame.shape[0], frame.shape[1])
        
        # Calculate metrics
        frame, self.hip_positions, metrics = calculate_metrics(
            keypoint_coords, keypoint_confs, frame, 0.4, self.hip_positions, 
            self.left_gait_detector, self.right_gait_detector, current_time, self.mode
        )

        # Display gait phases and cadence
        display_gait_phases(frame, metrics['left_gait_phase'], metrics['right_gait_phase'], metrics['cadence'])

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
            self.metrics_logger.close()
            self.video_recorder.stop_recording()
            print("\nPost-processing options:")
            self.video_recorder.post_recording_options()
            self.metrics_logger.post_logging_options()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Running Analysis using MoveNet")
    parser.add_argument('--model', type=str, default=THUNDER_PATH, help="Path to the MoveNet model")
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyzer = RunningAnalyzer(args.model)
    analyzer.run()

if __name__ == "__main__":
    main()