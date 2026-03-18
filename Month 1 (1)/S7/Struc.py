from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Defining the API components individually (The hard way)
print("--- 1. Individual API Calls ---")
# Step A: The Transformer
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fit() finds mean/std, transform() applies it
X_test_scaled = scaler.transform(X_test)       # ONLY transform() on test data to prevent data leakage

# Step B: The Estimator/Predictor
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)               # The model learns
predictions = clf.predict(X_test_scaled)       # The model predicts

print(f"Accuracy using individual steps: {clf.score(X_test_scaled, y_test):.3f}\n")

# 3. Using a Pipeline (The standard, elegant Scikit-Learn way)
print("--- 2. Pipeline API ---")
# The Pipeline combines Transformers and an Estimator into one object that shares the exact same API.
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # Transformer
    ('classifier', LogisticRegression())       # Estimator/Predictor
])

# You only call fit() once. The pipeline automatically calls fit_transform() on the scaler, 
# then fit() on the classifier.
pipeline.fit(X_train, y_train)

# You only call predict() once. The pipeline automatically transforms the test data 
# before passing it to the classifier.
pipeline_preds = pipeline.predict(X_test)

print(f"Accuracy using Pipeline: {pipeline.score(X_test, y_test):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, pipeline_preds, target_names=['Malignant', 'Benign']))