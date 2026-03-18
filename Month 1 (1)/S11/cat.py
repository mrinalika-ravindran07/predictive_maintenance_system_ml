
#This demonstrates how to convert text data into numbers so the Decision Tree algorithm (CART) can process it.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Create a dummy dataset
data = {
    'Fruit': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana', 'Orange'],
    'Color': ['Red', 'Yellow', 'Green', 'Orange', 'Yellow', 'Orange'],
    'Target_Is_Tasty': [1, 1, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

print("--- Original Data ---")
print(df.head())

# Define features (X) and target (y)
X = df[['Fruit', 'Color']]
y = df['Target_Is_Tasty']

# --- PREPROCESSING ---
# We use OneHotEncoder to turn "Apple", "Banana", "Red" into binary columns
# 'passthrough' tells it to leave other numerical columns alone (if we had any)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Fruit', 'Color'])
    ],
    remainder='passthrough' 
)

# Apply transformation
X_encoded = preprocessor.fit_transform(X)

# Get new column names for demonstration
new_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(['Fruit', 'Color'])

print("\n--- Encoded Data (What the Model Sees) ---")
# Convert sparse matrix to DataFrame for readability
X_display = pd.DataFrame(X_encoded.toarray(), columns=new_columns)
print(X_display)

# TRAIN MODEL
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)
print("\nModel trained successfully on encoded data!")