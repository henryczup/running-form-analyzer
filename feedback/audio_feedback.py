import pyttsx3
from threading import Thread
import time

class AudioFeedbackProvider:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.feedback_queue = []
        self.last_feedback_time = {}
        self.feedback_thread = Thread(target=self._feedback_loop)
        self.feedback_thread.daemon = True
        self.feedback_thread.start()

    def _feedback_loop(self):
        while True:
            if self.feedback_queue:
                feedback = self.feedback_queue.pop(0)
                self.engine.say(feedback)
                self.engine.runAndWait()
            time.sleep(0.1)

    def add_feedback(self, message: str, cooldown: int = 10):
        current_time = time.time()
        if message not in self.last_feedback_time or (current_time - self.last_feedback_time[message]) > cooldown:
            self.feedback_queue.append(message)
            self.last_feedback_time[message] = current_time

    def stop(self):
        self.engine.stop()
