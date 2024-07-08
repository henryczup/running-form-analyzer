import argparse
from core.config import THUNDER_PATH, Config
from core.analyzer import Analyzer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Running Analysis using various pose estimation models")
    parser.add_argument('--model_type', type=str, default='blazepose', choices=['movenet', 'blazepose', 'lite_hrnet'], help="Type of pose estimation model to use")
    # parser.add_argument('--model_path', type=str, default=THUNDER_PATH, help="Path to the model (for MoveNet)")
    parser.add_argument('--side', type=str, default='left', choices=['left', 'right'], help="Side of the body to analyze")
    # parser.add_argument('--filter_type', type=str, default='kalman', choices=['temporal', 'kalman', 'none'], help="Type of filter to use for foot strike detection")
    # parser.add_argument('--detection_axis', type=str, default='x', choices=['x', 'y'], help="Axis to use for foot strike detection (x: horizontal, y: vertical)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = Config.from_args(args)
    analyzer = Analyzer(config)
    analyzer.run()
    
if __name__ == "__main__":
    main()