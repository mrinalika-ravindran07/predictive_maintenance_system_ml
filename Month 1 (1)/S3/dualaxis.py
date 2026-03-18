#Goal: Plotting two datasets with different scales (e.g., Price vs. Volume) on the same chart.


import matplotlib.pyplot as plt
import numpy as np

# Data
months = np.arange(1, 13)
temperature = [30, 32, 35, 38, 40, 42, 39, 36, 33, 31, 29, 28] # Celsius
rainfall = [10, 12, 5, 2, 0, 0, 150, 200, 180, 60, 20, 10]      # mm

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: Temperature on Left Axis (ax1)
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (°C)', color=color)
ax1.plot(months, temperature, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second axes that shares the same x-axis
ax2 = ax1.twinx()  

# Plot 2: Rainfall on Right Axis (ax2)
color = 'tab:blue'
ax2.set_ylabel('Rainfall (mm)', color=color)  
ax2.bar(months, rainfall, color=color, alpha=0.3) # alpha makes it transparent
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Weather Data: Temperature vs Rainfall")
plt.show()