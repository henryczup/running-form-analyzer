import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pipeline = dai.Pipeline()

colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setPreviewSize(640, 480)
colorCam.setInterleaved(False)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
colorCam.preview.link(xoutRgb.input)

# def calculate_angle(a, b, c):
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End

#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)

#     if angle > 180.0:
#         angle = 360 - angle

#     return angle


def calculate_trunk_angle(shoulder, hip):
    trunk_line = hip - shoulder
    vertical_line = np.array([0, 1])  # Vertical line vector (0, 1)
    
    dot_product = np.dot(trunk_line, vertical_line)
    magnitudes = np.linalg.norm(trunk_line) * np.linalg.norm(vertical_line)
    
    angle_rad = np.arccos(dot_product / magnitudes)
    angle_deg = np.degrees(angle_rad)
    
    # Determine the orientation of the trunk line
    if trunk_line[0] < 0:
        angle_deg = -angle_deg
    
    return angle_deg


# def calculate_stride_rate(foot_strikes, duration):
#     stride_rate = len(foot_strikes) * 2 / (duration / 60)
#     return stride_rate

# def determine_foot_strike_pattern(ankle, heel, toe):
#     ankle_to_heel = np.linalg.norm(np.array(ankle) - np.array(heel))
#     ankle_to_toe = np.linalg.norm(np.array(ankle) - np.array(toe))
    
#     if ankle_to_heel < ankle_to_toe:
#         return "Heel Strike"
#     elif ankle_to_heel > ankle_to_toe:
#         return "Forefoot Strike"
#     else:
#         return "Midfoot Strike"

# def calculate_knee_angle(hip, knee, ankle):
#     thigh = np.array(knee) - np.array(hip)
#     lower_leg = np.array(ankle) - np.array(knee)
    
#     dot_product = np.dot(thigh, lower_leg)
#     magnitudes = np.linalg.norm(thigh) * np.linalg.norm(lower_leg)
    
#     angle_rad = np.arccos(dot_product / magnitudes)
#     angle_deg = np.degrees(angle_rad)
    
#     return angle_deg

# def calculate_arm_swing_angle(shoulder, elbow, vertical_axis):
#     upper_arm = np.array(elbow) - np.array(shoulder)
#     vertical_line = np.array(vertical_axis)
    
#     dot_product = np.dot(upper_arm, vertical_line)
#     magnitudes = np.linalg.norm(upper_arm) * np.linalg.norm(vertical_line)
    
#     angle_rad = np.arccos(dot_product / magnitudes)
#     angle_deg = np.degrees(angle_rad)
    
#     return angle_deg

# def calculate_vertical_oscillation(hip_positions):
#     max_position = np.max(hip_positions)
#     min_position = np.min(hip_positions)
#     vertical_oscillation = (max_position - min_position) / 2
#     return vertical_oscillation

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    foot_strikes = []
    hip_positions = []
    start_time = time.time()

    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)  # Pass the RGB image to pose.process()

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Trunk angle calculation
            left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
            left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
            right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            hip_midpoint = (left_hip + right_hip) / 2
            trunk_angle = calculate_trunk_angle(shoulder_midpoint, hip_midpoint)
            
            # # Stride rate calculation
            # left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            # right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            # if len(foot_strikes) == 0 or (left_heel[1] > 0.5 and right_heel[1] > 0.5):
            #     foot_strikes.append(time.time())
            # duration = time.time() - start_time
            # stride_rate = calculate_stride_rate(foot_strikes, duration)
            
            # # Foot strike pattern determination
            # left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            # left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            # left_toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            # foot_strike_pattern = determine_foot_strike_pattern(left_ankle, left_heel, left_toe)
            
            # # Knee angle calculation
            # left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            # knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)
            
            # # Arm swing angle calculation
            # left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            # arm_swing_angle = calculate_arm_swing_angle(left_shoulder, left_elbow, vertical_axis)
            
            # # Vertical oscillation calculation
            # hip_positions.append(hips[1])
            # if len(hip_positions) > 10:
            #     hip_positions.pop(0)
            # vertical_oscillation = calculate_vertical_oscillation(hip_positions)
            
            # Display metrics on the image
            cv2.putText(image, f"Trunk Angle: {trunk_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            # cv2.putText(image, f"Stride Rate: {stride_rate:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f"Foot Strike Pattern: {foot_strike_pattern}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f"Knee Angle: {knee_angle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f"Arm Swing Angle: {arm_swing_angle:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(image, f"Vertical Oscillation: {vertical_oscillation:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except:
            print('No poses detected')
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        height, width, _ = image.shape
        resized_image = cv2.resize(image, (int(width * 1.5), int(height * 1.5)))

        # Display the resized image
        cv2.imshow('rgb', resized_image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()