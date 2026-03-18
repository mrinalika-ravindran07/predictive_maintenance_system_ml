from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# 2. Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model 1: Underfitting (Too simple)
# max_depth=1 means it can only make one decision/split
model_underfit = DecisionTreeClassifier(max_depth=1, random_state=42)
model_underfit.fit(X_train, y_train)

print("--- Underfitting (max_depth=1) ---")
print(f"Training Accuracy: {accuracy_score(y_train, model_underfit.predict(X_train)):.3f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, model_underfit.predict(X_test)):.3f}\n")

# 4. Model 2: Overfitting (Too complex)
# max_depth=None means it will grow until all leaves are pure (memorization)
model_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
model_overfit.fit(X_train, y_train)

print("--- Overfitting (max_depth=None) ---")
print(f"Training Accuracy: {accuracy_score(y_train, model_overfit.predict(X_train)):.3f} (Memorized!)")
print(f"Testing Accuracy:  {accuracy_score(y_test, model_overfit.predict(X_test)):.3f} (Failed to generalize)\n")

# 5. Model 3: Good Fit (Balanced)
model_balanced = DecisionTreeClassifier(max_depth=5, random_state=42)
model_balanced.fit(X_train, y_train)

print("--- Good Fit (max_depth=5) ---")
print(f"Training Accuracy: {accuracy_score(y_train, model_balanced.predict(X_train)):.3f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, model_balanced.predict(X_test)):.3f}")