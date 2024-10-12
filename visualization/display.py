import cv2

def display_angles(frame, angles):
    y_offset = 30
    line_spacing = 25
    font_size = 0.5
    font_thickness = 1
    
    def put_text(text):
        nonlocal y_offset
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)
        y_offset += line_spacing

    for angle_name, angle_value in angles.items():
        put_text(f"{angle_name}: {angle_value:.2f}")


def display_metrics(frame, metrics, side):
    y_offset = 30
    line_spacing = 25  # Reduced from 30 to accommodate more lines
    font_size = 0.5  # Reduced from 0.7 for smaller text
    font_thickness = 1  # Reduced from 2 for thinner text
    
    def put_text(text):
        nonlocal y_offset
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness)
        y_offset += line_spacing

    put_text(f"Head Angle: {metrics['head_angle']:.2f} ({metrics['head_angle_assessment']})")
    put_text(f"Trunk Angle: {metrics['trunk_angle']:.2f} ({metrics['trunk_angle_assessment']})")

    put_text(f"Left Knee Angle at strike: {metrics['left_knee_angle_at_strike']:.2f} ({metrics['left_knee_assessment']})")
    put_text(f"Right Knee Angle at strike: {metrics['right_knee_angle_at_strike']:.2f} ({metrics['right_knee_assessment']})")
    put_text(f"Left Hip-Ankle Angle at strike: {metrics['left_hip_ankle_angle_at_strike']:.2f} ({metrics['left_hip_ankle_angle_assessment']})")
    put_text(f"Right Hip-Ankle Angle at strike: {metrics['right_hip_ankle_angle_at_strike']:.2f} ({metrics['right_hip_ankle_angle_assessment']})")
    
    if side == 'left':
        put_text(f"Left Elbow Angle: {metrics['left_elbow_angle']:.2f} ({metrics['left_elbow_angle_assessment']})")
        put_text(f"Left Shank Angle at strike: {metrics['left_shank_angle_at_strike']:.2f} ({metrics['left_shank_angle_assessment']})")
        put_text(f"Max Left Arm Forward Swing: {metrics['max_left_arm_forward_swing']:.2f} ({metrics['left_arm_forward_swing_assessment']})")
        put_text(f"Max Left Arm Backward Swing: {metrics['max_left_arm_backward_swing']:.2f} ({metrics['left_arm_backward_swing_assessment']})")
        put_text(f"Max Left Hip Forward Swing: {metrics['max_left_hip_forward_swing']:.2f} ({metrics['left_hip_forward_swing_assessment']})")
        put_text(f"Max Left Hip Backward Swing: {metrics['max_left_hip_backward_swing']:.2f} ({metrics['left_hip_backward_swing_assessment']})")
    else:
        put_text(f"Right Elbow Angle: {metrics['right_elbow_angle']:.2f} ({metrics['right_elbow_angle_assessment']})")
        put_text(f"Right Shank Angle at strike: {metrics['right_shank_angle_at_strike']:.2f} ({metrics['right_shank_angle_assessment']})")
        put_text(f"Max Right Arm Forward Swing: {metrics['max_right_arm_forward_swing']:.2f} ({metrics['right_arm_forward_swing_assessment']})")
        put_text(f"Max Right Arm Backward Swing: {metrics['max_right_arm_backward_swing']:.2f} ({metrics['right_arm_backward_swing_assessment']})")
        put_text(f"Max Right Hip Forward Swing: {metrics['max_right_hip_forward_swing']:.2f} ({metrics['right_hip_forward_swing_assessment']})")
        put_text(f"Max Right Hip Backward Swing: {metrics['max_right_hip_backward_swing']:.2f} ({metrics['right_hip_backward_swing_assessment']})")

    put_text(f"Vertical Oscillation: {metrics['vertical_oscillation']:.2f} cm ({metrics['vertical_oscillation_assessment']})")
    put_text(f"Left Foot Strike: {metrics['left_foot_strike']}")
    put_text(f"Right Foot Strike: {metrics['right_foot_strike']}")
    put_text(f"Steps per Minute: {metrics['steps_per_minute']:.2f}")
    put_text(f"Elapsed Time: {metrics['elapsed_time']:.2f} s")

def display_recommendations(frame, metrics):
    y_offset = 30
    line_spacing = 40
    font_size = 0.7
    font_thickness = 2

    recommendations = metrics['recommendations']

    for recommendation in recommendations:
        cv2.putText(frame, recommendation, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)
        y_offset += line_spacing

    if not recommendations:
        cv2.putText(frame, "Great form! Keep it up!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), font_thickness)


def display_mode(frame, metrics, angles, display_mode='metrics', side='left'):
    if display_mode == 'angles':
        display_angles(frame, angles)
    elif display_mode == 'metrics':
        display_metrics(frame, metrics, side)
    elif display_mode == 'recommendations':
        display_recommendations(frame, metrics)
    else:
        raise ValueError(f"Invalid display mode: {display_mode}")

    return frame