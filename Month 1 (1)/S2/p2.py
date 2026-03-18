#iloc: Uses Integer position (like Python lists).

#loc: Uses Labels (Index names and Column names).

import pandas as pd

data = {
    'Math': [85, 90, 78, 92],
    'Science': [80, 95, 82, 96]
}
# Setting custom index IDs to make the difference clear
df = pd.DataFrame(data, index=['ID_101', 'ID_102', 'ID_103', 'ID_104'])

print("--- DataFrame with Custom Index ---")
print(df)

# --- USING LOC (Label Based) ---
print("\n--- 1. .loc uses names (Select ID_102, Science column) ---")
print(df.loc['ID_102', 'Science']) 

print("\n--- 2. .loc Slicing is INCLUSIVE (Includes the end label) ---")
# Note: ID_103 IS included here
print(df.loc['ID_101':'ID_103'])

# --- USING ILOC (Position Based) ---
print("\n--- 3. .iloc uses positions (Row 1, Col 1) ---")
# Row 1 is ID_102, Col 1 is Science
print(df.iloc[1, 1])

print("\n--- 4. .iloc Slicing is EXCLUSIVE (Excludes the end index) ---")
# Note: Index 3 is NOT included here
print(df.iloc[0:3])