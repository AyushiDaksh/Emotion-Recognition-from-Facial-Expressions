import matplotlib.pyplot as plt
import pandas as pd

# Load the data
data = pd.read_csv("performance_log.csv")

# Plotting
plt.figure(figsize=(10, 6))

# FPS Plot
plt.subplot(2, 1, 1)
plt.plot(data['Timestamp'], data['FPS'], label='FPS')
plt.xlabel('Time (s)')
plt.ylabel('FPS')
plt.title('Frame Rate over Time')
plt.legend()

# Latency Plot
plt.subplot(2, 1, 2)
plt.plot(data['Timestamp'], data['Latency (s)'], label='Latency', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Latency (s)')
plt.title('Latency over Time')
plt.legend()

plt.tight_layout()
plt.show()

# CPU and GPU Utilization Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Timestamp'], data['CPU Utilization (%)'], label='CPU Usage', color='green')
plt.plot(data['Timestamp'], data['GPU Utilization (%)'], label='GPU Usage', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('CPU and GPU Utilization over Time')
plt.legend()
plt.show()