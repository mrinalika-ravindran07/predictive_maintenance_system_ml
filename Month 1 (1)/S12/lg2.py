import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# STEP 1: Generate Data ---
# We create a dataset with 10 features (some informative, some noise)
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5,
    random_state=42
)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

#  STEP 2: Initialize & Train LightGBM ---
print("Training LightGBM Model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

lgb_model.fit(X_train, y_train)

# STEP 3: Evaluation ---
y_pred = lgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# --- STEP 4: Visualizations ---

# Set up the figure size for two plots side-by-side
plt.figure(figsize=(15, 6))

# GRAPH 1: Feature Importance (Bar Chart)
plt.subplot(1, 2, 1)
# Create a DataFrame for importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('LightGBM Feature Importance')
plt.xlabel('Importance Score')

# GRAPH 2: Confusion Matrix (Heatmap)
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show the plots
plt.tight_layout()
plt.show()