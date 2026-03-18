import pandas as pd

# Table 1: Student Demographics
students = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Amit', 'Bhavna', 'Chirag', 'Deepti']
})

# Table 2: Exam Results (Note: ID 3 is missing, ID 5 is new)
scores = pd.DataFrame({
    'ID': [1, 2, 4, 5],
    'Math': [85, 90, 92, 88]
})

print("--- 1. Inner Join (Intersection) ---")
# Only students present in BOTH tables (ID 1, 2, 4)
# Chirag (3) is lost (no score). ID 5 is lost (no name).
merged_inner = pd.merge(students, scores, on='ID', how='inner')
print(merged_inner)

print("\n--- 2. Left Join (Preserve Left Side) ---")
# Keep all Students. If score is missing, put NaN.
# Chirag (3) stays, but gets NaN for Math.
merged_left = pd.merge(students, scores, on='ID', how='left')
print(merged_left)

print("\n--- 3. Outer Join (Union) ---")
# Keep EVERYONE. 
# Chirag (3) has no score. ID 5 has no Name. All are kept.
merged_outer = pd.merge(students, scores, on='ID', how='outer')
print(merged_outer)