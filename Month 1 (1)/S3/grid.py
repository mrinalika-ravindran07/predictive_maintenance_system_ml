#Goal: Managing multiple subplots effectively using array indexing.


import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(x/3)

# Create a grid: 2 rows, 2 columns
# figsize controls the overall canvas size
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# axs is now a 2x2 numpy array of Axes objects. 
# We access them like a matrix: axs[row, col]

# Top Left
axs[0, 0].plot(x, y1, 'tab:blue')
axs[0, 0].set_title('Sine')

# Top Right
axs[0, 1].plot(x, y2, 'tab:orange')
axs[0, 1].set_title('Cosine')

# Bottom Left (limiting y-axis to avoid ugly tan spikes)
axs[1, 0].plot(x, y3, 'tab:green')
axs[1, 0].set_ylim(-5, 5)
axs[1, 0].set_title('Tangent')

# Bottom Right
axs[1, 1].plot(x, y4, 'tab:red')
axs[1, 1].set_title('Exponential')

# Automatically adjust spacing between plots so labels don't overlap
plt.tight_layout()
plt.show()