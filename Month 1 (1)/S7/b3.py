from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 1. DATA PREPARATION
# Generate a synthetic binary classification dataset
# 100 samples, 2 useful features
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, n_clusters_per_class=1)
# Logistic Regression is the standard baseline for classification
model = LogisticRegression()

# 3. FIT
print("Training Logistic Regression...")
model.fit(X, y)

# 4. PREDICT (Two ways)
# New sample to predict
new_sample = [[0.5, 0.5]]

# Way A: Hard Prediction (Class 0 or 1)
class_pred = model.predict(new_sample)

# Way B: Soft Prediction (Probability of being class 0 vs class 1)
prob_pred = model.predict_proba(new_sample)

# 5. INSPECT
print(f"Input features: {new_sample}")
print(f"Predicted Class: {class_pred[0]}")
print(f"Confidence (Probability): {prob_pred[0]}")
print(f"  -> {prob_pred[0][0]*100:.2f}% chance of Class 0")
print(f"  -> {prob_pred[0][1]*100:.2f}% chance of Class 1")