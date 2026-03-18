#Lab Objective
#Induce Overfitting: Create a model with too much complexity (high-degree polynomial) for the amount of data available.

#Diagnose: Visualize the "Generalization Gap" (Train vs. Test error).

#Remedy: Apply Ridge (L2) and Lasso (L1) with automated hyperparameter tuning.

#Advanced Visualization: Generate "Coefficient Trace Plots" to see exactly how regularization kills noise.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
n_samples = 40

# True function: Sin wave
X = np.sort(np.random.rand(n_samples))
y_true = np.cos(1.5 * np.pi * X)

# Observed data: True function + heavy noise
y = y_true + np.random.randn(n_samples) * 0.3
X = X[:, np.newaxis]

# Split data (Critical for detecting overfitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  2. Feature Engineering: "Exploding" the complexity ---
# We create a 15-degree polynomial. For a simple curve, this is massive overkill.
degree = 15
poly = PolynomialFeatures(degree=degree, include_bias=False)
scaler = StandardScaler()

# Transform Train and Test data
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale (MANDATORY for Regularization)
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

print(f"Original Feature count: 1")
print(f"Exploded Feature count: {X_train_scaled.shape[1]}")