
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Generate Data
np.random.seed(42)
n_samples = 100
X = np.sort(np.random.rand(n_samples, 1), axis=0)
y = np.cos(1.5 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Polynomial Features                                                                       
poly = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Standardize Features
scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)
# --- 3. Model Training --
# Model A: Unregularized Linear Regression (The Overfit Model)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Model B: Ridge (L2) with Built-in Cross-Validation
# alphas: list of regularization strengths to try
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13)) 
ridge.fit(X_train_scaled, y_train)

# Model C: Lasso (L1) with Built-in Cross-Validation
lasso = LassoCV(n_alphas=100, cv=5, random_state=42, max_iter=100000)
lasso.fit(X_train_scaled, y_train)

# --- 4. Evaluation ---
models = {'Linear (OLS)': lr, 'Ridge (L2)': ridge, 'Lasso (L1)': lasso}
results = []

for name, model in models.items():
    # Calculate MSE
    train_mse = mean_squared_error(y_train, model.predict(X_train_scaled))
    test_mse = mean_squared_error(y_test, model.predict(X_test_scaled))
    
    # Store results
    results.append({
        'Model': name,
        'Train MSE': train_mse, 
        'Test MSE': test_mse,
        'Gap (Overfitting)': test_mse - train_mse,
        'Optimal Alpha': getattr(model, 'alpha_', 0) # LR has no alpha
    })

results_df = pd.DataFrame(results).round(4)

print("\n--- Performance Report ---")
print("Note the huge gap in OLS (Low Train Error, High Test Error)")
print(results_df)

# --- 5. Visualization of the Curves ---
X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
X_plot_poly = poly.transform(X_plot)
X_plot_scaled = scaler.transform(X_plot_poly)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, color='red', marker='x', label='Test Data')
plt.plot(X_plot, np.cos(1.5 * np.pi * X_plot), 'k--', label='True Underlying Function', alpha=0.3)

colors = {'Linear (OLS)': 'red', 'Ridge (L2)': 'blue', 'Lasso (L1)': 'green'}
styles = {'Linear (OLS)': ':', 'Ridge (L2)': '-', 'Lasso (L1)': '-'}

for name, model in models.items():
    y_plot = model.predict(X_plot_scaled)
    plt.plot(X_plot, y_plot, color=colors[name], linestyle=styles[name], linewidth=2, label=f"{name}")

plt.title("Visualizing Overfitting vs. Regularization")
plt.ylim(-2, 2)
plt.legend()
plt.show()