#Concept: Understanding how to grab specific columns by name and specific rows by their numerical position.

import pandas as pd

# 1. Create a sample dataset
data = {
    'Student': ['Amit', 'Bhavna', 'Chirag', 'ira', 'Esha'],
    'Math': [85, 90, 78, 92, 88],
    'Science': [80, 95, 82, 96, 79],
    'English': [75, 85, 80, 88, 90]
}

df = pd.DataFrame(data)

print("--- Original DataFrame ---")
print(df)
print("\n")

# 2. Column Selection (Indexing)
print("--- 1. Selecting a Single Column (Returns Series) ---")
math_scores = df['Math']
print(math_scores)

print("\n--- 2. Selecting Multiple Columns (Returns DataFrame) ---")
subset = df[['Student', 'Science']]
print(subset)

# 3. Row Selection by Position (.iloc)
# Syntax: df.iloc[row_index, column_index]
print("\n--- 3. Selecting the First Row (Index 0) ---")
print(df.iloc[0])

print("\n--- 4. Slicing Rows (0 to 2 - Index 3 is excluded) ---")
print(df.iloc[0:3])