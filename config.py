
# Known parameters
KNOWN_TORSO_LENGTH_CM = 52  # Average height of a person in cm

# Camera parameters for your XPS 15 
HFOV_DEG = 74  # Horizontal field of view in degrees
IMAGE_WIDTH_PX = 1280  # Image width in pixels

# Download models folder in directory
THUNDER_PATH = "models/thunder-float32.tflite"
LIGHTNING_PATH = "models/lightning-float32.tflite"


config = {
    'lite_hrnet': {
        'num_joints': 17,
        'input_size': (256, 256),
        'output_channels': 17,
        'stage_channels': [40, 80, 160],
        'stage_blocks': [2, 4, 2],
        'downsample_rate': 4
    }
}