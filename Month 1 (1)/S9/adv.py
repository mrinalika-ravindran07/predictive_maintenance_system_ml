import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV

# 1. Generate Data with Many Features
X, y = make_regression(n_samples=100, n_features=20, n_informative=5, noise=0.1, random_state=42)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Standardize Features
scaler = StandardScaler()
# : Actually create the scaled variables
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Setup alphas from very small (no regularization) to large (high regularization)
alphas = np.logspace(-4, 0.5, 100)
coefs = []

# Loop through alphas and record coefficients (Lasso Path)
for a in alphas:
    lasso_temp = Lasso(alpha=a)
    lasso_temp.fit(X_train_scaled, y_train)
    coefs.append(lasso_temp.coef_)


lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42).fit(X_train_scaled, y_train)

# 3. Plotting
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the coefficient paths
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficients')
plt.title('Lasso Coefficients as a Function of Alpha')
plt.axis('tight')

plt.axvline(lasso_cv.alpha_, linestyle='--', color='k', label=f'Optimal Alpha (CV) = {lasso_cv.alpha_:.4f}')

plt.legend()
plt.show()