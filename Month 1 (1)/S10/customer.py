import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. GENERATE SYNTHETIC CUSTOMER DATA
# We simulate 1000 customers
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 70, n_samples),
    'monthly_bill': np.random.randint(20, 100, n_samples),
    'customer_support_calls': np.random.randint(0, 10, n_samples),
    'contract_length_months': np.random.randint(1, 24, n_samples)
}

df = pd.DataFrame(data)

# Create a 'Churn' label based on some logic (plus some random noise)
# Logic: High bills + High support calls = High chance of Churn
churn_prob = (
    (df['monthly_bill'] > 80).astype(int) + 
    (df['customer_support_calls'] > 5).astype(int) + 
    np.random.rand(n_samples) # Add noise so it's not perfect
)
df['churn'] = (churn_prob > 1.5).astype(int) 

print("--- Customer Data Snapshot ---")
print(df.head())

# 2. PREPARE DATA
X = df[['age', 'monthly_bill', 'customer_support_calls', 'contract_length_months']]
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. BUILD BASELINE MODEL
# Logistic Regression is the standard 'baseline'
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. EVALUATE
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBaseline Model Accuracy: {accuracy:.2f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

coeffs = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print("\n--- Feature Importance (Coefficients) ---")
print(coeffs.sort_values(by='Coefficient', ascending=False))
print("(Positive values increase risk of Churn, Negative values decrease it)")