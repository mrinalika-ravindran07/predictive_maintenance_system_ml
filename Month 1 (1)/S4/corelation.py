import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


x = np.linspace(-10, 10, 100)
y = x**2 

corr_parabola = np.corrcoef(x, y)[0, 1]

print(f"--- The Parabola Trap ---")
print(f"Correlation between X and X^2: {corr_parabola:.2f}")
print("Logic: Because the relationship is symmetrical, the 'up' cancels out the 'down'.\n")

# 2. Anscombe's Quartet (The Classic 'Don't Trust Stats' Dataset)
anscombe = sns.load_dataset("anscombe")

print("--- Anscombe's Quartet Stats ---")
# Showing that means and variances are nearly identical across all 4 sets
stats = anscombe.groupby("dataset").agg(["mean", "std"])
print(stats)

# Showing that correlations are also identical (~0.816)
correlations = anscombe.groupby("dataset")[['x','y']].corr().iloc[0::2,-1]
print("\nCorrelations (r):\n", correlations)

# 3. Visualizing why the stats lied
sns.set_theme(style="ticks")
g = sns.lmplot(
    x="x", y="y", col="dataset", hue="dataset", data=anscombe,
    col_wrap=2, ci=None, palette="muted", height=4, scatter_kws={"s": 50, "alpha": 1}
)

# Adding a title to the figure
g.fig.suptitle("Anscombe's Quartet: Same Stats, Different Realities", fontsize=16)
plt.tight_layout()
plt.show()