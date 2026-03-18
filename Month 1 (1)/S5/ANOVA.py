#Scenario: Manufacturing Quality Control. Goal: Show why you cannot just run 3 separate t-tests (The Multiple Comparison Problem)
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import scipy.stats as stats
import numpy as np  
from statsmodels.stats.multicomp import pairwise_tukeyhsd
np.random.seed(10)
factory_a = np.random.normal(100, 5, 100)
factory_b = np.random.normal(100, 5, 100)
factory_c = np.random.normal(103, 5, 100) 

data = pd.DataFrame({
    'weight': np.concatenate([factory_a, factory_b, factory_c]),
    'factory': ['A']*100 + ['B']*100 + ['C']*100
})


f_stat, p_val = stats.f_oneway(factory_a, factory_b, factory_c)

print(f"ANOVA F-statistic: {f_stat:.2f}")
print(f"ANOVA P-value: {p_val:.5f}")

if p_val < 0.05:
    print("\n--- Significant Difference Found! Locating the culprit... ---")
    tukey = pairwise_tukeyhsd(endog=data['weight'], groups=data['factory'], alpha=0.05)
    print(tukey)
else:
    print("No significant difference found.")