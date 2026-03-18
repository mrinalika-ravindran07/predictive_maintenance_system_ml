#This program manually implements Batch Gradient Descent. It visualizes the Mean Squared Error (MSE) landscape in 3D and shows the "path" the algorithm takes to find the minimum global cost.


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_cost(X, y, theta):
    """
    Compute MSE: J(theta) = (1/2m) * sum((h_theta(x) - y)^2)
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    
    for i in range(iterations):
        prediction = X.dot(theta)
        # Gradient Calculation: (1/m) * X.T * (prediction - y)
        gradients = (1/m) * X.T.dot(prediction - y)
        
        # Update Weights
        theta = theta - learning_rate * gradients
        
        # Log History
        theta_history[i,:] = theta.T
        cost_history[i] = compute_cost(X, y, theta)
        
    return theta, cost_history, theta_history

# --- Data Generation (Simple Regression for 3D Viz) ---
X_raw = 2 * np.random.rand(100, 1)
y = 4 + 3 * X_raw + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X_raw] # Add bias

# --- Training ---
lr = 0.1
n_iter = 50
theta_init = np.random.randn(2,1)
theta_final, cost_hist, theta_hist = gradient_descent(X_b, y, theta_init, lr, n_iter)

# --- 3D Visualization of Cost Function ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid for 3D surface
ms = np.linspace(theta_final[0]-2, theta_final[0]+2, 20)
bs = np.linspace(theta_final[1]-2, theta_final[1]+2, 20)
M, B = np.meshgrid(ms, bs)
zs = np.array([compute_cost(X_b, y, np.array([[m], [b]])) 
               for m, b in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)

# Plot Surface and Descent Path
ax.plot_surface(M, B, Z, cmap='viridis', alpha=0.6)
ax.plot(theta_hist[:,0], theta_hist[:,1], cost_hist, color='r', marker='o', 
        label='Gradient Descent Path', zorder=10)

ax.set_xlabel('Intercept (Theta 0)')
ax.set_ylabel('Slope (Theta 1)')
ax.set_zlabel('Cost (MSE)')
ax.set_title('Gradient Descent Optimization Path minimizing MSE')
plt.legend()
plt.show()