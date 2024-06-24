import os
import cv2
from datetime import datetime
from video_player import play_video

class VideoRecorder:
    def __init__(self, output_dir='videos'):
        self.output_dir = output_dir
        self.video_writer = None
        self.recording = False
        self.frame_size = None
        self.output_filename = None

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def start_recording(self, frame):
        if not self.recording:
            self.frame_size = (frame.shape[1], frame.shape[0])
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_filename = f"{self.output_dir}/run_video_{current_time}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, 30.0, self.frame_size)
            self.recording = True
            print(f"Started recording: {self.output_filename}")

    def record_frame(self, frame):
        if self.recording and self.video_writer is not None:
            self.video_writer.write(frame)

    def stop_recording(self):
        if self.recording:
            self.video_writer.release()
            self.recording = False
            print(f"Stopped recording: {self.output_filename}")

    def __del__(self):
        if self.recording:
            self.stop_recording()

    def post_recording_options(self):
        print("\nRecording completed.")
        while True:
            choice = input("Options:\n1. Rewatch video\n2. Save video with new name\n3. Exit (Delete if not saved)\nEnter your choice (1/2/3): ")
            
            if choice == '1':
                play_video(self.output_filename)
                print("\nVideo playback completed. Returning to options.")
                continue  # This ensures we go back to the start of the while loop
            elif choice == '2':
                self.save_video_with_new_name()
            elif choice == '3':
                if self.output_filename and os.path.exists(self.output_filename):
                    confirm = input(f"Video will be deleted. Are you sure you want to exit? (y/n): ")
                    if confirm.lower() == 'y':
                        os.remove(self.output_filename)
                        print(f"Video deleted: {self.output_filename}")
                break
            else:
                print("Invalid choice. Please try again.")

    def save_video_with_new_name(self):
        new_filename = input("Enter new filename (or press Enter to keep current name): ")
        if new_filename:
            new_path = os.path.join(self.output_dir, new_filename)
            os.rename(self.output_filename, new_path)
            self.output_filename = new_path
            print(f"Video saved as: {self.output_filename}")
        else:
            print(f"Video saved as: {self.output_filename}")