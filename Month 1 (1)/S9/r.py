#Regularization: Ridge (L2) vs. Lasso (L1)

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# 1. Generate Data (Sine wave with noise)
np.random.seed(42)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = np.cos(1.5 * np.pi * X) + np.random.randn(n_samples) * 0.1
X = X[:, np.newaxis]

# 2. Create a High-Degree Polynomial (Degree 15) to force Overfitting
degree = 15
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)


scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# 3. Train Models
# A. No Regularization (Standard Linear Regression)
model_no_reg = LinearRegression()
model_no_reg.fit(X_poly_scaled, y)

# B. Ridge (L2) - Alpha controls strength (Higher alpha = flatter line)
model_ridge = Ridge(alpha=0.1)
model_ridge.fit(X_poly_scaled, y)

# C. Lasso (L1)
model_lasso = Lasso(alpha=0.01) # Lasso often needs smaller alpha
model_lasso.fit(X_poly_scaled, y)

# 4. Visualization
X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
X_plot_poly = poly.transform(X_plot)
X_plot_scaled = scaler.transform(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='black', label='Data')
plt.plot(X_plot, model_no_reg.predict(X_plot_scaled), color='red', label='No Regularization (Overfit)')
plt.plot(X_plot, model_ridge.predict(X_plot_scaled), color='blue', linewidth=2, label='Ridge (L2)')
plt.plot(X_plot, model_lasso.predict(X_plot_scaled), color='green', linestyle='--', linewidth=2, label='Lasso (L1)')

plt.title(f"Regularization Effects on High-Degree ({degree}) Polynomial")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()