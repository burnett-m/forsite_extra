import psutil
import time
import datetime
import csv

outputFile = r"E:\_Models\CPU_RAM_monitor.csv"

def monitor_cpu_ram(interval_seconds=1):
    while True:
        # Get CPU usage as a percentage
        cpu_percent = psutil.cpu_percent(interval=interval_seconds)

        # Get RAM usage as a percentage
        ram_percent = psutil.virtual_memory().percent

        # Get time
        now = datetime.datetime.now()

        # Print the results
        CPUmonitor = f"CPU Usage: {cpu_percent:.2f}% | RAM Usage: {ram_percent:.2f}% | {now}"
        print(CPUmonitor)
        CPUmonitorL = [CPUmonitor]

        with open(outputFile,'a', newline='') as myFile:
            writer = csv.writer(myFile)
            writer.writerow(CPUmonitorL)
            
        # Wait for the specified interval
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Set the monitoring interval (in seconds)
    monitoring_interval = 5
    monitor_cpu_ram(interval_seconds=monitoring_interval)


