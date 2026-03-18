#Prove that Mean and Standard Deviation are fragile, while Median and IQR are robust. We will inject "poison" (extreme outliers) into a healthy dataset and watch the statistics break.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
clean_data = np.random.normal(loc=170, scale=10, size=100) # Mean=170, Std=10

# 2. Inject "Poison" (3 data errors: e.g., someone wrote 1700cm instead of 170cm)
poison_data = np.append(clean_data, [1700, 1800, 5000])

def analyze_spread(data, label):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    
    # Interquartile Range (IQR) - The Robust Spread
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    
    print(f"--- {label} ---")
    print(f"Mean: {mean:.2f} (Shift: {mean - 170:.2f})")
    print(f"Median: {median:.2f} (Shift: {median - 170:.2f})")
    print(f"Std Dev: {std_dev:.2f}")
    print(f"IQR: {iqr:.2f}\n")

analyze_spread(clean_data, "CLEAN DATA")
analyze_spread(poison_data, "POISONED DATA")

# Visualizing the destruction
plt.figure(figsize=(10, 4))
sns.boxplot(x=poison_data)
plt.title("Boxplot Showing Extreme Outliers")
plt.show()