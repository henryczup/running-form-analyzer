import cv2
import numpy as np
import time
import tensorflow as tf
from config import THUNDER_PATH
from detection import extract_keypoints
from measurements import calculate_metrics
from visualization import draw_connections, draw_keypoints


def main():
    interpreter = tf.lite.Interpreter(model_path=THUNDER_PATH)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)

    hip_positions = []
    left_ankle_velocities = [] 
    left_ankle_positions = []
    right_ankle_positions = []
    start_time = time.time()
    timestamps = []
    frame_count = 0
    

    mode = "dev"  # Default mode is dev mode

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time() - start_time

        # Resize the frame to 192x192 for the MoveNet model
        img = frame.copy()
        img = tf.image.resize(tf.expand_dims(img, axis=0), (256,256))
        input_image = tf.cast(img, dtype=tf.float32)

        # Set up input and output 
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Draw the keypoints and connections on the frame
        draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.3)
        draw_connections(frame, keypoints_with_scores, confidence_threshold=0.3)

        # Extract keypoint coordinates and confidence scores
        keypoint_coords, keypoint_confs = extract_keypoints(keypoints_with_scores, 840, 1280)
        frame, hip_positions, left_ankle_positions, left_ankle_velocities = calculate_metrics(keypoint_coords, keypoint_confs, frame, 0.4, hip_positions, left_ankle_positions, right_ankle_positions, mode, left_ankle_velocities)

        # # Get foot positions
        # if keypoint_confs[6] > .3:
        #     left_ankle_positions.append(keypoint_coords[6][1])
        #     timestamps.append(current_time)
        # else:
        #     left_ankle_positions.append(0)
        #     timestamps.append(current_time)


        cv2.imshow("frame", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            mode = "dev"
        elif key == ord('u'):
            mode = "user"

    cv2.destroyAllWindows()
    # visualize_cadence(left_ankle_positions, right_ankle_positions, frame_count)

if __name__ == "__main__":
    main()