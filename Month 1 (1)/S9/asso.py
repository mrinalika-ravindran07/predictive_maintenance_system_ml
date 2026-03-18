#The Concept: Lasso (L1) has a unique property: it zeroes out the coefficients of irrelevant features. This makes it excellent for automatic feature selection.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
    

# 1. Setup Data with 10 features, but only 3 are useful
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=10, n_informative=3, noise=10, random_state=42)

feature_names = [f"Feature_{i}" for i in range(10)]

# 2. Train Models
# Standard Linear Regression
lr = LinearRegression()
lr.fit(X, y)

# Lasso Regression
lasso = Lasso(alpha=1.0) # Alpha determines how aggressive the selection is
lasso.fit(X, y)

# 3. Compare Coefficients
df_coeffs = pd.DataFrame({
    'Feature': feature_names,
    'Linear Reg Coeffs': lr.coef_,
    'Lasso Coeffs': lasso.coef_
})

# Round for readability
df_coeffs = df_coeffs.round(2)

print("--- Feature Selection Demonstration ---")
print("Notice how Lasso sets irrelevant feature weights to exactly 0.0")
print("-" * 50)
print(df_coeffs)

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(lr.coef_, marker='o', label='Linear Regression')
plt.plot(lasso.coef_, marker='x', label='Lasso (L1)', color='red', markersize=10)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Coefficient Comparison: Linear vs Lasso")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Magnitude")
plt.legend()
plt.show()