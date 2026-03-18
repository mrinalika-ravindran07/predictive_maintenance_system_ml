#Modifying data using slicing and optimized scalar access. at and iat are faster than loc and iloc when accessing a single cell
import pandas as pd

df = pd.DataFrame({
    'Student': ['A', 'B', 'C', 'D'],
    'Score': [85, 90, 78, 92]
})

# 1. Conditional Slicing & Assignment
print("--- Original ---")
print(df)

print("\n--- Give 5 bonus marks to scores below 80 ---")
# Locate rows where Score < 80, select 'Score' column, add 5
df.loc[df['Score'] < 80, 'Score'] += 5
print(df)

# 2. Fast Scalar Access (at/iat)
print("\n--- Using .at for fast single value modification ---")
# .at[RowLabel, ColLabel]
df.at[0, 'Score'] = 100 
print(f"Modified Student A score to: {df.at[0, 'Score']}")

print("\n--- Using .iat for fast position modification ---")
# .iat[RowPosition, ColPosition]
df.iat[1, 1] = 99 
print(df)