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