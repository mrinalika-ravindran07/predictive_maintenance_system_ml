import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42) 

# Define sample sizes
n_control = 1000 
n_variant = 1000  #

p_control = 0.10
p_variant = 0.12
control_data = np.random.binomial(n=1, p=p_control, size=n_control)
variant_data = np.random.binomial(n=1, p=p_variant, size=n_variant)

# Create a DataFrame for easier handling
df = pd.DataFrame({
    'Group': ['Control'] * n_control + ['Variant'] * n_variant,
    'Conversion': np.concatenate([control_data, variant_data])
})

print("--- Data Preview ---")
print(df.sample(5))
print("\n")

group_stats = df.groupby('Group')['Conversion'].agg(['mean', 'std', 'count'])
print("--- Conversion Rates by Group ---")
print(group_stats)
print(f"\nDifference in means: {group_stats.loc['Variant', 'mean'] - group_stats.loc['Control', 'mean']:.4f}")


t_stat, p_val = stats.ttest_ind(variant_data, control_data, equal_var=False)

print("\n--- T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value (Two-sided): {p_val:.4f}")


alpha = 0.05 

print("\n--- Conclusion ---")
if p_val < alpha:
    print