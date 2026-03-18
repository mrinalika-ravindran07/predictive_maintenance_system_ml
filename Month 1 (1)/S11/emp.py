import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Create a mock Employee Attrition Dataset
data = {
    'Age': [25, 34, 45, 29, 31, 22, 40, 55, 33, 28],
    'Department': ['Sales', 'R&D', 'Sales', 'HR', 'R&D', 'Sales', 'HR', 'R&D', 'Sales', 'R&D'],
    'DailyRate': [110, 800, 1200, 300, 500, 150, 900, 1100, 400, 600],
    'JobSatisfaction': [1, 4, 3, 2, 4, 1, 3, 4, 2, 3], # Scale 1-4
    'Attrition': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'] # Target
}
df = pd.DataFrame(data)

# 2. Preprocessing (Handling Categorical Data)
# 'Department' is text, so we must encode it to numbers
le = LabelEncoder()
df['Department_Encoded'] = le.fit_transform(df['Department'])

# Define Features (X) and Target (y)
feature_cols = ['Age', 'DailyRate', 'JobSatisfaction', 'Department_Encoded']
X = df[feature_cols]
y = df['Attrition'] # Target: Yes/No

# 3. Split Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Decision Tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 5. Make Predictions
y_pred = clf.predict(X_test)

# 6. Evaluate Performance
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Check Feature Importance (Interpretability)
print("\n--- Feature Importance ---")
# Zip creates pairs of (Feature Name, Importance Score)
for feature, importance in zip(feature_cols, clf.feature_importances_):
    print(f"{feature}: {importance:.4f}")