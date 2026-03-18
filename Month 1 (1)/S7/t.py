import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# --- Step 1: Generate Synthetic Data (Sine Wave + Noise) ---
np.random.seed(0)
n_samples = 30

# True function: f(x) = cos(1.5 * pi * x)
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1 # Add random noise

# Reshape X for Scikit-Learn
X = X[:, np.newaxis]

# --- Step 2: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- Step 3: Visualize Different Complexities ---
plt.figure(figsize=(14, 5))
degrees = [1, 4, 15] # 1=Underfit, 4=Good Fit, 15=Overfit
titles = ['Underfitting (Degree 1)', 'Good Fit (Degree 4)', 'Overfitting (Degree 15)']

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    
    # Create a pipeline: Transform data to polynomial -> Linear Regression
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    # Plotting
    X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
    plt.plot(X_plot, pipeline.predict(X_plot), label="Model", color='red', linewidth=2)
    plt.plot(X_plot, true_fun(X_plot), label="True Function", color='green', linestyle='--')
    plt.scatter(X_train, y_train, edgecolor='b', s=20, label="Train Data")
    plt.scatter(X_test, y_test, edgecolor='r', s=20, label="Test Data", marker='x')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{titles[i]}\nTrain Score: {train_score:.2f}\nTest Score: {test_score:.2f}")
    plt.ylim(-2, 2)
    plt.legend(loc="best")

plt.tight_layout()
plt.show()