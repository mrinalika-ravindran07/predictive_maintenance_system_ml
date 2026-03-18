
#Goal: Visualizing how data is distributed across groups (Boxplots and Violin plots).

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('penguins')

# Create a figure with 2 subplots (using Matplotlib's OO system!)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Boxplot (Classic statistical summary)
sns.boxplot(
    data=df, 
    x="species", 
    y="body_mass_g", 
    hue="sex",
    palette="pastel",
    ax=axs[0]  # Tell Seaborn to draw on the first axes
)
axs[0].set_title("Boxplot: Quartiles & Outliers")

# Plot 2: Violin Plot (Distribution density + Boxplot)
# split=True combines the 'hue' categories into half-violins to save space
sns.violinplot(
    data=df, 
    x="species", 
    y="body_mass_g", 
    hue="sex", 
    split=True, 
    inner="quart", 
    palette="muted",
    ax=axs[1]
)
axs[1].set_title("Violin Plot: Density Distribution")

plt.tight_layout()
plt.show()