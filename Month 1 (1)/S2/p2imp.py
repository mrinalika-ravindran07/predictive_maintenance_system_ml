import pandas as pd
import numpy as np

# Data with duplicates and missing values
data = {
    'Student': ['Amit', 'Bhavna', 'Amit', 'Deepti', 'Deepti'],
    'Score': [85, 90, 85, np.nan, 92] # Note: Amit is repeated exactly
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df)

# --- HANDLING DUPLICATES ---
print("\n--- 1. Remove Duplicate Rows ---")
# keep='first' keeps the first occurrence and deletes the rest
df = df.drop_duplicates(keep='first')
print(df)

# --- SMART FILLING ---
print("\n--- 2. Fill Missing Score with the Average (Mean) ---")
average_score = df['Score'].mean()
print(f"Average Score is: {average_score}")

# Fill NaNs with the calculated mean
df['Score'] = df['Score'].fillna(average_score)
print(df)