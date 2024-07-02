
# Known parameters
KNOWN_TORSO_LENGTH_CM = 52  # Average height of a person in cm

# Camera parameters for your XPS 15 
HFOV_DEG = 74  # Horizontal field of view in degrees
IMAGE_WIDTH_PX = 1280  # Image width in pixels

# Download models folder in directory
THUNDER_PATH = "models/thunder-float32.tflite"
LIGHTNING_PATH = "models/lightning-float32.tflite"

# edges for the pose graph
EDGES = {(0,1): 'm', 
         (0,2): 'c', 
         (1,3): 'm', 
         (2,4): 'c', 
         (0,5): 'm', 
         (0,6): 'c', 
         (5,7): 'm', 
         (7,9): 'm', 
         (6,8): 'c', 
         (8,10): 'c', 
         (5,6): 'y', 
         (5,11): 'm', 
         (6,12): 'c', 
         (11,12): 'y', 
         (11,13): 'm', 
         (13,15): 'm', 
         (12,14): 'c', 
         (14,16): 'c'
         }