import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# 1. Generate a non-linear dataset (a sine wave with some noise)
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.5 # Add noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define polynomial degrees to test
degrees = [1, 4, 15]
labels = ["High Bias, Low Variance", "Sweet Spot", "Low Bias, High Variance"]

print("--- Bias-Variance Tradeoff Demonstration ---")
for degree, label in zip(degrees, labels):
    # Create a pipeline that first creates polynomial features, then fits a linear model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate Mean Squared Error (MSE)
    train_error = mean_squared_error(y_train, model.predict(X_train))
    test_error = mean_squared_error(y_test, model.predict(X_test))
    
    print(f"\nDegree: {degree} ({label})")
    print(f"Training Error (Proxy for Bias):     {train_error:.3f}")
    print(f"Testing Error  (Proxy for Variance): {test_error:.3f}")