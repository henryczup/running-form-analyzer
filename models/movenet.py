# movenet.py

import tensorflow as tf
import numpy as np

class MoveNetModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, image):
        input_image = tf.cast(tf.image.resize(tf.expand_dims(image, axis=0), (256, 256)), dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]