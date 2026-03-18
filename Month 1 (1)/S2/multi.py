
#Working with DataFrames that have more than one index (Hierarchical Indexing). This is common in advanced data analysis.

import pandas as pd

# Creating a Multi-Index DataFrame
arrays = [
    ['Class A', 'Class A', 'Class B', 'Class B'],
    ['Amit', 'Bhavna', 'Chirag', 'Deepti']
]
index = pd.MultiIndex.from_arrays(arrays, names=('Class', 'Student'))

df = pd.DataFrame({'Score': [88, 92, 79, 95]}, index=index)

print("--- Multi-Index DataFrame ---")
print(df)

# 1. Outer Level Slicing
print("\n--- Select all students in Class A ---")
print(df.loc['Class A'])

# 2. Inner Level Slicing (Specific Tuple)
print("\n--- Select Deepti from Class B ---")
print(df.loc[('Class B', 'Deepti')])

# 3. Cross-Section (.xs)
# Useful to select 'Bhavna' regardless of which class she is in (if names were unique or repeated)
# Or selecting the second level index
print("\n--- Advanced: Select all students named 'Chirag' from any class ---")
# level=1 refers to the 'Student' index
print(df.xs('Chirag', level='Student'))