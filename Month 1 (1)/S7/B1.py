import numpy as np
from sklearn.linear_model import LinearRegression


X = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([2, 4, 6, 8, 10])          


model = LinearRegression()

print("Training model...")
model.fit(X, y)

X_new = np.array([[6], [7]])
predictions = model.predict(X_new)

# 5. INSPECT
print(f"Prediction for input 6: {predictions[0]}")
print(f"Prediction for input 7: {predictions[1]}")
print(f"Learned Coefficient (Slope): {model.coef_[0]}")
print(f"Learned Intercept: {model.intercept_}")