import csv
from typing import Dict, Any
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_dir: str = 'tests/logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file = os.path.join(log_dir, f'run_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self.csv_file = None
        self.csv_writer = None
        self.metrics = []

    def initialize_logging(self, available_metrics: Dict[str, Any]):
        self.metrics = list(available_metrics.keys())
        
        # Initialize CSV file with all metrics
        self.csv_file = open(self.log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp'] + self.metrics)

    def log_metrics(self, timestamp: float, metrics: Dict[str, Any]):
        if not self.metrics:
            print("Error: Metrics have not been initialized. Call initialize_logging() first.")
            return

        row = [timestamp] + [metrics.get(metric, '') for metric in self.metrics]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written immediately

    def close(self):
        if self.csv_file:
            self.csv_file.close()

    def post_logging_options(self):
        print("\nMetrics logging completed.")
        while True:
            choice = input("Options:\n1. View log summary\n2. Rename log file\n3. Exit\nEnter your choice (1/2/3): ")
            
            if choice == '1':
                self.view_log_summary()
            elif choice == '2':
                self.rename_log_file()
                break  # Exit the loop after renaming
            elif choice == '3':
                self.delete_log_file()
                break
            else:
                print("Invalid choice. Please try again.")

    def view_log_summary(self):
        if not self.metrics:
            print("No metrics were logged.")
            return

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
        averages = []
        for i in range(1, len(headers)):
            try:
                avg = sum(float(row[i]) for row in data if row[i]) / len([row for row in data if row[i]])
                averages.append(avg)
            except ValueError:
                averages.append(None)  # For non-numeric data
        
        for header, avg in zip(headers[1:], averages):
            if avg is not None:
                print(f"Average {header}: {avg:.2f}")
            else:
                print(f"{header}: Non-numeric data")

    def rename_log_file(self):
        new_filename = input("Enter new filename (or press Enter to keep current name): ")
        if new_filename:
            new_path = os.path.join(self.log_dir, new_filename)
            os.rename(self.log_file, new_path)
            self.log_file = new_path
            print(f"Log file saved as: {self.log_file}")
        else:
            print(f"Log file saved as: {self.log_file}")
    
    def delete_log_file(self):
        if self.log_file and os.path.exists(self.log_file):
            confirm = input(f"Do you want your log file to be deleted? (y/n): ")
            if confirm.lower() == 'y':
                os.remove(self.log_file)
                print(f"Log file deleted: {self.log_file}")
