#Goal: Completely changing the look by manipulating "spines" (the borders). This is often used for cleaner, minimalist, or scientific aesthetic.
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = x**2

fig, ax = plt.subplots()

ax.plot(x, y)

# Move left and bottom spines to x = 0 and y = 0 (Center the axes)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')

# Eliminate upper and right axes spines (borders)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_title("Centered Spines (Math Textbook Style)", y=1.02)

plt.show()