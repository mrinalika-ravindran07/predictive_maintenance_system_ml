import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 1: Data Generation
# We generate a "harder" dataset to make the tuning necessary and visible.
# weights=[0.7, 0.3] creates a slightly imbalanced dataset (70% class 0, 30% class 1)
print("Generating complex synthetic data...")
X, y = make_classification(
    n_samples=2000, 
    n_features=20, 
    n_informative=10, # More informative features make the problem complex
    n_redundant=5, 
    weights=[0.7, 0.3], 
    random_state=42
)

feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 2: Baseline Model (The "Before")
print("\n--- Phase 1: Training Baseline Model ---")
# Standard Random Forest with default settings
base_rf = RandomForestClassifier(random_state=42)
base_rf.fit(X_train, y_train)

base_pred = base_rf.predict(X_test)
base_acc = accuracy_score(y_test, base_pred)
print(f"Baseline Accuracy (Default Params): {base_acc:.4f}")

# STEP 3: Hyperparameter Tuning (The "Improvement")
print("\n--- Phase 2: Running Grid Search (Optimizing) ---")
print("Testing combinations of parameters... This may take a moment.")

# Define the "Grid" of settings to test
# n_estimators: Number of trees
# max_depth: How deep each tree can grow (prevents overfitting)
# min_samples_split: Minimum data points required to split a node
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# GridSearchCV tries EVERY combination to find the winner
# cv=3 means "Cross Validation" (splits training data 3 ways to verify results)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1, # Uses all computer cores for speed
    verbose=1
)

grid_search.fit(X_train, y_train)

# Get the best model from the search
best_rf = grid_search.best_estimator_

# STEP 4: Evaluation & Comparison
print("\n--- Phase 3: Final Results ---")
print(f"Best Parameters Found: {grid_search.best_params_}")

tuned_pred = best_rf.predict(X_test)
tuned_acc = accuracy_score(y_test, tuned_pred)

print(f"Baseline Accuracy: {base_acc:.4f}")
print(f"Tuned Accuracy:    {tuned_acc:.4f}")
print(f"Improvement:       {(tuned_acc - base_acc) * 100:.2f}%")

# STEP 5: Visualizing the Improvement
plt.figure(figsize=(12, 5))

# Plot 1: Accuracy Comparison
plt.subplot(1, 2, 1)
sns.barplot(x=['Baseline', 'Tuned'], y=[base_acc, tuned_acc], palette=['gray', 'green'])
plt.ylim(0, 1.0)
plt.title('Model Accuracy Improvement')
plt.ylabel('Accuracy Score')
for i, v in enumerate([base_acc, tuned_acc]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

# Plot 2: Confusion Matrix of the Best Model
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, tuned_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()