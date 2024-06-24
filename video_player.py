import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.playback_speed = 1.0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Video Player')
        self.setGeometry(100, 100, 800, 600)

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # Slider for seeking
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frame_count - 1)
        self.slider.sliderMoved.connect(self.slider_moved)

        # Buttons
        self.play_pause_button = QPushButton('Play')
        self.play_pause_button.clicked.connect(self.play_pause)

        self.rewind_button = QPushButton('Rewind')
        self.rewind_button.clicked.connect(self.rewind)

        # Playback speed dropdown
        self.speed_combo = QComboBox()
        speeds = ['0.25x', '0.5x', '0.75x', '1x', '1.25x', '1.5x', '2x']
        self.speed_combo.addItems(speeds)
        self.speed_combo.setCurrentText('1x')
        self.speed_combo.currentTextChanged.connect(self.change_playback_speed)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.rewind_button)
        button_layout.addWidget(self.play_pause_button)
        button_layout.addWidget(QLabel('Speed:'))
        button_layout.addWidget(self.speed_combo)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.slider)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.slider.setValue(self.current_frame)

    def next_frame(self):
        self.current_frame += 1
        if self.current_frame >= self.frame_count:
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.show_frame()

    def play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_pause_button.setText('Play')
        else:
            self.timer.start(int(1000 / (self.fps * self.playback_speed)))
            self.play_pause_button.setText('Pause')
        self.playing = not self.playing

    def rewind(self):
        self.current_frame = max(0, self.current_frame - int(self.fps * 5))  # Rewind 5 seconds
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        self.show_frame()

    def change_playback_speed(self, speed_text):
        self.playback_speed = float(speed_text[:-1])  # Remove 'x' and convert to float
        if self.playing:
            self.timer.setInterval(int(1000 / (self.fps * self.playback_speed)))

    def slider_moved(self, position):
        self.current_frame = position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.show_frame()

    def closeEvent(self, event):
        self.cap.release()
        event.accept() 

def play_video(video_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    player = VideoPlayer(video_path)
    player.show()
    app.exec_()

    return

if __name__ == "__main__":
    play_video("path_to_your_video.mp4")