import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class ExactLinearRegression:
    def __init__(self):
        self.weights = None
        
    def fit(self, X, y):
        # Add intercept term (column of 1s) to X (Bias trick)
        # Shape becomes (m, n+1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # The Normal Equation: theta = (X.T * X)^-1 * X.T * y
        # This provides the exact analytical solution minimizing MSE
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)

# Simulation: Multiple Linear Regression ---
# Generating complex data: 1000 samples, 5 features, noise added
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Custom Model
model = ExactLinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- 2. Visualization (Projecting 5D data to 1D for viewing) ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='blue', edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3, label='Perfect Fit Line')
plt.title(f'Multiple Linear Regression (5 Features)\nModel Weights: {np.round(model.weights, 2)}')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

print("This program demonstrates that Simple and Multiple regression utilize the same")
print("Matrix equation: θ = (XᵀX)⁻¹Xᵀy. The 'weights' array contains the coefficients.")