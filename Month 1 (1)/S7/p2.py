#: Splitting a Dataset and Checking for Data Leakage


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset with a specific mean and standard deviation
np.random.seed(42)
X = np.random.normal(loc=50.0, scale=10.0, size=(1000, 5)) # 1000 samples, 5 features
y = (X[:, 0] + X[:, 1] > 100).astype(int)                  # Simple target variable

print("--- The WRONG Way (Data Leakage) ---")
# Mistake: Fitting the scaler on the ENTIRE dataset before splitting
leaky_scaler = StandardScaler()
X_leaky_scaled = leaky_scaler.fit_transform(X)

# The split happens AFTER scaling. 
# The test set's mean and variance have "leaked" into the training data's scaling!
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_leaky_scaled, y, test_size=0.2, random_state=42
)

# Proof of Leakage: The scaler used test data to calculate its mean. 
# Notice the training set mean is NOT exactly 0, because the scaler was influenced by the test set.
print(f"Leaky Train Set Mean: {np.mean(X_train_leak):.6f} (Should be exactly 0 if scaled independently)")
print(f"Leaky Test Set Mean:  {np.mean(X_test_leak):.6f}\n")


print("--- The RIGHT Way (No Leakage) ---")
# 1. Split the data FIRST
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Fit the scaler ONLY on the training data
clean_scaler = StandardScaler()
X_train_clean_scaled = clean_scaler.fit_transform(X_train_clean)

# 3. Transform the test data using the parameters learned strictly from the training data
X_test_clean_scaled = clean_scaler.transform(X_test_clean)

# Proof of No Leakage: The training set mean is exactly 0 (or computationally close to it).
# The test set is shifted based ONLY on training parameters.
print(f"Clean Train Set Mean: {np.mean(X_train_clean_scaled):.6f} (Correctly isolated)")
print(f"Clean Test Set Mean:  {np.mean(X_test_clean_scaled):.6f} (Transformed purely on train stats)")