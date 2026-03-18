import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(101)

n = 100
control = np.random.normal(loc=45, scale=10, size=n)   #
variant_a = np.random.normal(loc=48, scale=10, size=n) 
variant_b = np.random.normal(loc=55, scale=12, size=n) 

df = pd.DataFrame({
    'TimeOnSite': np.concatenate([control, variant_a, variant_b]),
    'Layout': (['Control'] * n) + (['Variant A'] * n) + (['Variant B'] * n)
})

print("--- Data Snapshot ---")
print(df.groupby('Layout')['TimeOnSite'].describe().round(2))
print("\n")



print("--- Assumption Checks ---")


stat, p_levene = stats.levene(control, variant_a, variant_b)
if p_levene > 0.05:
    print(f" Variances are equal (Levene p={p_levene:.3f})")
else:
    print(f" Variances differ (Levene p={p_levene:.3f}) - Consider Welch's ANOVA")


model = ols('TimeOnSite ~ C(Layout)', data=df).fit()
stat, p_shapiro = stats.shapiro(model.resid)

if p_shapiro > 0.05:
    print(f"Residuals are normal (Shapiro p={p_shapiro:.3f})")
else:
    print(f" Residuals may not be normal (Shapiro p={p_shapiro:.3f})")


f_stat, p_val = stats.f_oneway(control, variant_a, variant_b)

print("\n--- ANOVA Results ---")
print(f"F-Statistic: {f_stat:.4f}")
print(f"P-Value:     {p_val:.4e}") # Scientific notation for very small numbers

anova_table = sm.stats.anova_lm(model, typ=2) 
ss_between = anova_table['sum_sq'][0]
ss_total = anova_table['sum_sq'].sum()
eta_sq = ss_between / ss_total

print(f"Effect Size (Eta Squared): {eta_sq:.4f}")
if eta_sq > 0.14: print("(Large effect)")
elif eta_sq > 0.06: print("(Medium effect)")
else: print("(Small effect)")



if p_val < 0.05:
    print("\n--- Tukey's HSD Post-Hoc Test ---")
    tukey = pairwise_tukeyhsd(endog=df['TimeOnSite'], groups=df['Layout'], alpha=0.05)
    print(tukey)
else:
    print("\nNo significant difference found; Post-hoc testing not required.")

plt.figure(figsize=(10, 6))

# Boxplot shows the quartiles
sns.boxplot(x='Layout', y='TimeOnSite', data=df, palette='pastel', showfliers=False)

# Swarmplot overlays individual points to show density/distribution
sns.swarmplot(x='Layout', y='TimeOnSite', data=df, color='black', alpha=0.5, size=3)

plt.title('Time on Site Distribution by Layout (ANOVA Analysis)')
plt.ylabel('Time on Site (seconds)')
plt.show()