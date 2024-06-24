import csv
from typing import Dict
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file = os.path.join(log_dir, f'run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'trunk_angle', 'knee_angle', 'arm_swing_angle', 
                                  'distance_cm', 'vertical_oscillation', 'left_hip_ankle_angle', 
                                  'right_hip_ankle_angle', 'cadence', 'fps'])

    def log_metrics(self, timestamp: float, metrics: Dict[str, float]):
        self.csv_writer.writerow([timestamp, metrics['trunk_angle'], metrics['knee_angle'], 
                                  metrics['arm_swing_angle'], metrics['distance_cm'], 
                                  metrics['vertical_oscillation'], metrics['left_hip_ankle_angle'], 
                                  metrics['right_hip_ankle_angle'], metrics['cadence'], metrics['fps']])
        self.csv_file.flush()  # Ensure data is written immediately

    def close(self):
        self.csv_file.close()

    def post_logging_options(self):
        print("\nMetrics logging completed.")
        while True:
            choice = input("Options:\n1. View log summary\n2. Rename log file\n3. Exit (Delete if not renamed)\nEnter your choice (1/2/3): ")
            
            if choice == '1':
                self.view_log_summary()
            elif choice == '2':
                self.rename_log_file()
                break  # Exit the loop after renaming
            elif choice == '3':
                if self.log_file and os.path.exists(self.log_file):
                    confirm = input(f"Do you want your log file to be deleted? (y/n): ")
                    if confirm.lower() == 'y':
                        os.remove(self.log_file)
                        print(f"Log file deleted: {self.log_file}")
                break
            else:
                print("Invalid choice. Please try again.")

    def view_log_summary(self):
        with open(self.log_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip the header row
            data = list(reader)
        
        if not data:
            print("No data recorded in the log file.")
            return

        print("\nLog Summary:")
        print(f"Total records: {len(data)}")
        print(f"Time range: {data[0][0]} to {data[-1][0]}")
        
        # Calculate averages for numeric columns
        averages = [sum(float(row[i]) for row in data) / len(data) for i in range(1, len(headers))]
        
        for header, avg in zip(headers[1:], averages):
            print(f"Average {header}: {avg:.2f}")

    def rename_log_file(self):
        new_filename = input("Enter new filename (or press Enter to keep current name): ")
        if new_filename:
            new_path = os.path.join(self.log_dir, new_filename)
            os.rename(self.log_file, new_path)
            self.log_file = new_path
            print(f"Log file saved as: {self.log_file}")
        else:
            print(f"Log file saved as: {self.log_file}")