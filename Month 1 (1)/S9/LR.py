#The Concept: Linear regression assumes a straight-line relationship ($y = mx + c$). If data is curved, we need to add "powers" of existing features ($x^2, x^3$) to capture the complexity.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. Generate "Curved" Data (Quadratic)
np.random.seed(42)
X = 2 - 3 * np.random.normal(0, 1, 30)
y = X - 2 * (X ** 2) + np.random.normal(-3, 3, 30)
X = X[:, np.newaxis] # Reshape for sklearn

# 2. Fit Models
# Model A: Standard Linear Regression (Underfitting)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Model B: Polynomial Regression (Degree 2)
# We create a pipeline: first transform features, then apply linear regression
poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_reg.fit(X, y)

# 3. Visualization
X_seq = np.linspace(X.min(), X.max(), 100).reshape(-1, 1) # Smooth line for plotting

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Noisy Data')
plt.plot(X_seq, lin_reg.predict(X_seq), color='red', linestyle='--', label='Linear Fit (Degree 1)')
plt.plot(X_seq, poly_reg.predict(X_seq), color='blue', linewidth=2, label='Polynomial Fit (Degree 2)')

plt.title("Linear vs Polynomial Regression")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()
plt.show()