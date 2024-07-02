from collections import deque
import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_estimate_error, measurement_noise, process_noise):
        self.state = initial_state
        self.estimate_error = initial_estimate_error
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise

    def update(self, measurement):
        # Prediction
        predicted_state = self.state
        predicted_estimate_error = self.estimate_error + self.process_noise

        # Update
        kalman_gain = predicted_estimate_error / (predicted_estimate_error + self.measurement_noise)
        self.state = predicted_state + kalman_gain * (measurement - predicted_state)
        self.estimate_error = (1 - kalman_gain) * predicted_estimate_error

        return self.state

class TemporalFilter:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.positions = deque(maxlen=window_size)

    def update(self, new_position: float) -> float:
        self.positions.append(new_position)
        if len(self.positions) < self.window_size:
            return new_position
        return np.mean(list(self.positions))

class Filters:
    @staticmethod
    def create_filter(filter_type: str, **kwargs):
        if filter_type == "kalman":
            return KalmanFilter(
                initial_state=kwargs.get('initial_state', 0),
                initial_estimate_error=kwargs.get('initial_estimate_error', 1),
                measurement_noise=kwargs.get('measurement_noise', 0.1),
                process_noise=kwargs.get('process_noise', 0.01)
            )
        elif filter_type == "temporal":
            return TemporalFilter(window_size=kwargs.get('window_size', 10))
        elif filter_type == "none":
            return None
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")