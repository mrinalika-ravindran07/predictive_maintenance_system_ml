import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Data
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5, # 5 features actually matter
    n_redundant=2,   # 2 features are just noise/copies
    random_state=42
)

# Convert to DataFrame for easier handling
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Random Forest
# n_estimators=100 means we create 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# 4. Predictions
y_pred_rf = rf_model.predict(X_test)

# 5. Evaluation
print("=== Random Forest Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))