import pandas as pd

# Dataset 1: Class A
df_class_A = pd.DataFrame({
    'Student': ['Amit', 'Bhavna'],
    'Score': [85, 90]
})

# Dataset 2: Class B
df_class_B = pd.DataFrame({
    'Student': ['Chirag', 'Deepti'],
    'Score': [78, 92]
})

# Dataset 3: Extra Info (Same students as Class A, different data)
df_extra_info = pd.DataFrame({
    'City': ['Delhi', 'Mumbai'],
    'Age': [15, 16]
})

print("--- Vertical Concatenation (Adding Rows) ---")
# Like gluing Class B to the bottom of Class A
all_students = pd.concat([df_class_A, df_class_B], ignore_index=True)
print(all_students)

print("\n--- Horizontal Concatenation (Adding Columns) ---")
# Like gluing Extra Info to the right side of Class A
# Warning: Indexes must match for this to work correctly!
profile = pd.concat([df_class_A, df_extra_info], axis=1)
print(profile)