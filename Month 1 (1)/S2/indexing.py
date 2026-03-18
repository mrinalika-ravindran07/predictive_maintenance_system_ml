#Boolean Indexing (Filtering)
#Concept: Selecting rows based on the data inside them, rather than their position. This is the "SQL WHERE clause" of Pandas.
import pandas as pd

df = pd.DataFrame({
    'Student': ['Amit', 'Bhavna', 'Chirag', 'Deepti', 'Esha'],
    'Math': [45, 90, 32, 92, 88],
    'Status': ['Fail', 'Pass', 'Fail', 'Pass', 'Pass']
})

print("--- Filtering Data ---")

# 1. Simple Condition
print("\n--- Who scored above 80 in Math? ---")
high_scorers = df[df['Math'] > 80]
print(high_scorers)

# 2. Multiple Conditions (AND / OR)
# Use & for AND, | for OR. Parentheses are mandatory!
print("\n--- Passed AND Math > 85 ---")
top_students = df[(df['Status'] == 'Pass') & (df['Math'] > 85)]
print(top_students)

# 3. The .isin() method (Like SQL 'IN')
print("\n--- Select specific students by list ---")
selected = df[df['Student'].isin(['Amit', 'Deepti'])]
print(selected)