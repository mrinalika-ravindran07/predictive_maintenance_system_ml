import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 1: Initialize LightGBM Classifier ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    verbose=-1  # Suppress warnings
)

#STEP 2: Train ---
lgb_model.fit(X_train, y_train)

#STEP 3: Predict ---
y_pred_lgb = lgb_model.predict(X_test)

# STEP 4: Evaluation ---
print("=== LightGBM Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb):.4f}")