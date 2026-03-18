#Scenario: A Pharmaceutical Clinical Trial.Goal: Prove that "statistical significance" ($p < 0.05$) is dangerous without looking at the Confidence Interval (the range of plausible truth).

import numpy as np
import scipy.stats as stats

np.random.seed(42)  # Ensures everyone gets the exact same "random" numbers
n = 50              #


placebo = np.random.normal(loc=100, scale=15, size=n)
drug    = np.random.normal(loc=102, scale=15, size=n)

# 2. Calculate Basic Statistics
diff_mean = np.mean(drug) - np.mean(placebo)
# We run a standard independent T-test (assuming equal variance for simplicity)
t_stat, p_val = stats.ttest_ind(drug, placebo)
var_drug = np.var(drug, ddof=1)
var_placebo = np.var(placebo, ddof=1)
se_diff = np.sqrt((var_drug / n) + (var_placebo / n))

# Step B: Find the "T-Critical" value (The multiplier for 95% confidence)
dof = n + n - 2                          
t_crit = stats.t.ppf(0.975, df=dof)       l

# Step C: Calculate the Margin of Error and the Bounds
margin_of_error = t_crit * se_diff
ci_low = diff_mean - margin_of_error
ci_high = diff_mean + margin_of_error

print(f"--- RESULTS ---")
print(f"P-value: {p_val:.4f}")
print(f"Difference in Means: {diff_mean:.2f}")
print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")

# 4. The Lesson: Interpreting the results
print("\n--- INTERPRETATION ---")
if p_val < 0.05:
    print("P-value Result: STATISTICALLY SIGNIFICANT")
else:
    print("P-value Result: NOT SIGNIFICANT")
    print("   -> (We failed to prove the difference exists. But does that mean it's zero?)")

if ci_low < 0 and ci_high > 0:
    print(f"CI Result: The Interval [{ci_low:.2f}, {ci_high:.2f}] CROSSES ZERO.")
    print("   -> We cannot rule out that the drug is harmful (negative) or useless (zero).")
    print("   -> The wide range shows our study was too 'noisy' to be sure.")