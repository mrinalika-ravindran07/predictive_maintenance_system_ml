import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# STEP 1: GET DATA
data = load_iris()
X = data.data
y = data.target

# Split the data into Training (X_train) and Testing sets
# This creates the variables you were missing!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 2: DEFINE MODEL ---
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss'  # Note: ensure there is an underscore (_) here
)

#  STEP 3: TRAIN MODE
# Now this will work because X_train exists
xgb_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

# --- STEP 4: EVALUATE MODEL ---
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("Model trained successfully!")