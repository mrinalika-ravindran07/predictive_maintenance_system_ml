#Goal: Manually placing an axes inside another axes to show a "zoom-in" effect. This demonstrates the true power of OO—you can put an axes anywhe

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
y = np.sin(x) * np.exp(-x/5) # Decaying sine wave

fig, ax = plt.subplots(figsize=(10, 6))

# Main plot
ax.plot(x, y, color='black')
ax.set_title("Decaying Sine Wave with Zoom Inset")

# Define the region for the inset axes
# [left, bottom, width, height] in percentages of the parent axes
# 0.5, 0.5 means start at 50% x and 50% y (middle of the plot)
left, bottom, width, height = [0.5, 0.5, 0.35, 0.35] 
ax_inset = ax.inset_axes([left, bottom, width, height])

# Plot the same data on the inset
ax_inset.plot(x, y, color='tab:red')

# Zoom in on a specific region (e.g., the first peak)
x1, x2, y1, y2 = 1.0, 2.0, 0.6, 0.9
ax_inset.set_xlim(x1, x2)
ax_inset.set_ylim(y1, y2)
ax_inset.set_title("Zoom on First Peak")

# Draw connectors to show where the zoom comes from
ax.indicate_inset_zoom(ax_inset, edgecolor="black")

plt.show()