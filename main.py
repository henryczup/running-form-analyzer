import cv2
import depthai as dai
import numpy as np
import time
import tensorflow as tf

from measurements import calculate_metrics, draw_connections, draw_keypoints, extract_keypoints, visualize_cadence

# Download models folder in directory
THUNDER_PATH = "models/thunder-float32.tflite"
LIGHTNING_PATH = "models/lightning-float32.tflite"

interpreter = tf.lite.Interpreter(model_path=THUNDER_PATH)
interpreter.allocate_tensors()


# # For oak 
# pipeline = dai.Pipeline()
# colorCam = pipeline.create(dai.node.ColorCamera)
# colorCam.setPreviewSize(640, 480)
# colorCam.setInterleaved(False)
# xoutRgb = pipeline.create(dai.node.XLinkOut)
# xoutRgb.setStreamName("rgb")
# colorCam.preview.link(xoutRgb.input)

# # For on device camera
# # Camera parameters for the IMX378 sensor
# FOCAL_LENGTH_MM = 4.81  # Effective focal length in mm
# HFOV_DEG = 69  # Horizontal field of view in degrees
# SENSOR_WIDTH_MM = 6.17  # Sensor width in mm for IMX378
# IMAGE_WIDTH_PX = 4056  # Image width in pixels for full resolution


cap = cv2.VideoCapture(0)

hip_positions = []
left_foot_positions = []
right_foot_positions = []
start_time = time.time()
frame_count = 0
total_fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

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
    frame, hip_positions = calculate_metrics(keypoint_coords, keypoint_confs, frame, frame_count, 0.4, hip_positions, left_foot_positions, right_foot_positions)

    # Get foot positions
    if keypoint_confs[6] > .3:
        left_foot_positions.append(keypoint_coords[6][1])
    else:
        left_foot_positions.append(0)
    
    if keypoint_confs[7] > .3:
        right_foot_positions.append(keypoint_coords[7][1])
    else:
        right_foot_positions.append(0)

    # make frame larger for better visualization
    cv2.imshow("frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

visualize_cadence(left_foot_positions, right_foot_positions, frame_count)