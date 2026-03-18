#Goal: Create a simple figure and axes explicitly, avoiding the plt.plot() state-machine.

import matplotlib.pyplot as plt
import numpy as np

# 1. Generate dummy data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 2. The Object-Oriented Setup
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Plotting on the Axes object
ax.plot(x, y, label='Sine Wave', color='blue', linewidth=2)

# 4. Setting attributes using setters
ax.set_title("Sine Wave Function")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()

# 5. Display
plt.show()