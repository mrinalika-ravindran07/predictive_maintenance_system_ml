#"Average" is meaningless without "Shape." We will use Q-Q Plots (an advanced visual tool) and the Jarque-Bera test to prove data isn't Normal.
import seaborn as sns
import scipy.stats as stats
import numpy as np              # Fixes the NameError for np.random
import matplotlib.pyplot as plt

# Generate 3 distinct datasets
d1 = np.random.normal(0, 1, 1000)             # Normal
d2 = np.random.exponential(1, 1000)           # Right Skewed (e.g., Call Center Wait Times)
d3 = np.random.laplace(0, 1, 1000)            # High Kurtosis (Fat Tails / Risk)

data_dict = {"Normal": d1, "Skewed": d2, "Fat Tails": d3}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, data) in enumerate(data_dict.items()):
    # Calculate Advanced Stats
    skew = stats.skew(data)
    kurt = stats.kurtosis(data) # Fisher's definition (Normal = 0)
    
    # Normality Test (Jarque-Bera)
    jb_stat, p_value = stats.jarque_bera(data)
    is_normal = "YES" if p_value > 0.05 else "NO"

    # Plot
    sns.histplot(data, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f"{name}\nSkew: {skew:.2f} | Kurt: {kurt:.2f}\nIs Normal? {is_normal}")
    
plt.tight_layout()
plt.show()

