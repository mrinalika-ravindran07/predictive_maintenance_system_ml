import pandas as pd
import numpy as np

# 1. Create a "Messy" DataFrame
data = {
    'Student': ['Amit', 'Bhavna', 'Chirag', 'Deepti', 'Esha'],
    'Math': [85, np.nan, 78, 92, np.nan],  # NaN indicates missing data
    'Science': [80, 95, np.nan, 96, 79],
    'City': ['Delhi', 'Mumbai', 'Delhi', np.nan, 'Bangalore']
}

df = pd.DataFrame(data)

print("--- Original Messy Data ---")
print(df)

# 2. Check for Nulls
print("\n--- 1. How many missing values per column? ---")
print(df.isnull().sum())

# 3. The 'Nuclear' Option: Drop any row with missing data
print("\n--- 2. Drop rows with ANY missing values ---")
df_clean = df.dropna()
print(df_clean)
# Note: This often deletes too much data!