#Key Concepts Covered: Linearity, Homoscedasticity, Normality of Residuals, Multicollinearity (VIF).


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Data Generation with intentional flaws to detect ---
np.random.seed(42)
X1 = np.random.rand(100)
X2 = 0.5 * X1 + np.random.normal(0, 0.1, 100) # High correlation with X1 (Multicollinearity)
X3 = np.random.rand(100)
y = 2 + 3*X1 + 4*X2 + 1.5*X3 + np.random.normal(0, 0.5, 100)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

# --- Model Fitting ---
X = df[['X1', 'X2', 'X3']]
X_with_const = sm.add_constant(X) # Add intercept
model = sm.OLS(df['y'], X_with_const).fit()
residuals = model.resid

# --- DIAGNOSTIC DASHBOARD ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression Diagnostic Dashboard', fontsize=16)

# 1. Linearity & Homoscedasticity: Residuals vs Fitted
# Ideal: Random scatter around horizontal line at 0. No "funnel" shape.
axes[0, 0].scatter(model.fittedvalues, residuals, alpha=0.7)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('1. Homoscedasticity Check (Resid vs Fit)')

# 2. Normality of Residuals: Q-Q Plot
# Ideal: Points falling on the 45-degree red line.
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('2. Normality Check (Q-Q Plot)')

# 3. Residual Histogram
axes[1, 0].hist(residuals, bins=15, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('3. Residual Distribution')

# 4. Multicollinearity Check: Variance Inflation Factor (VIF)
# Ideal: VIF < 5. High VIF indicates the variable is redundant.
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Plotting VIF as a bar chart
axes[1, 1].bar(vif_data["Feature"], vif_data["VIF"], color=['orange' if v > 5 else 'green' for v in vif_data["VIF"]])
axes[1, 1].axhline(5, color='red', linestyle='--', label='Threshold (5.0)')
axes[1, 1].set_title('4. Multicollinearity (VIF Score)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("Diagnostic Summary:")
print("-" * 30)
print(f"Normality (Shapiro-Wilk): p-value = {stats.shapiro(residuals)[1]:.4f} (p < 0.05 implies non-normal)")
print(f"High Multicollinearity detected in:\n{vif_data[vif_data['VIF'] > 5]}")