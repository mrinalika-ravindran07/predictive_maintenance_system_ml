#Goal: Show the "U-shaped" curve of error. As complexity increases, Bias decreases (we capture the trend), but Variance increases (we capture noise). The "Sweet Spot" is where the Test Error is lowest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Generate Data ---
np.random.seed(0)
n_samples = 50
X = np.sort(np.random.rand(n_samples))
true_y = np.cos(1.5 * np.pi * X)
y = true_y + np.random.randn(n_samples) * 0.2
X = X[:, np.newaxis]

# Split Data (50/50 split to exaggerate effects for small data)
split = int(n_samples * 0.5)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_errors = []
test_errors = []
degrees = range(1, 15)

# --- Loop through complexities ---
for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)
    
    # Record Error (MSE)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

# --- Plot the Tradeoff Curve ---
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Training Error (Bias proxy)', marker='o', color='blue')
plt.plot(degrees, test_errors, label='Testing Error (Variance proxy)', marker='o', color='orange')

plt.title('Bias-Variance Tradeoff: Model Complexity vs Error')
plt.xlabel('Polynomial Degree (Complexity)')
plt.ylabel('Mean Squared Error')
plt.xticks(degrees)
plt.legend()
plt.grid(True, alpha=0.3)

# Annotate the "Sweet Spot"
best_degree = degrees[np.argmin(test_errors)]
plt.axvline(x=best_degree, color='green', linestyle='--', label='Sweet Spot')
plt.text(best_degree + 0.5, max(test_errors)/2, f'Optimal Degree: {best_degree}', color='green')

plt.show()