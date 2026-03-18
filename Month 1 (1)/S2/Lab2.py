import pandas as pd
import numpy as np
import json

# DATASET 1: CUSTOMERS (Nested & Unstructured) ---
customers = pd.DataFrame({
    'Cust_ID': [101, 102, 103, 104, 105],
    
    'Address_JSON': [
        '{"City": "Delhi", "Zip": "110001"}', 
        '{"City": "Mumbai", "Zip": "400001"}', 
        '{"City": "delhi", "Zip": "110020"}',  # Lowercase issue
        np.nan,                                # Missing JSON
        '{"City": "Bangalore", "Zip": "560001"}'
    ],
    # Problem B: Phone number buried in text (needs Regex)
    'Contact_Info': [
        'Phone: 987-654-3210 (Home)', 
        'Mob: 9999999999', 
        'Call 888-888-8888', 
        'No number', 
        'Ph: 7776665555'
    ],
    'Join_Date': ['2023-01-01', '2023-02-15', '2023-01-10', '2023-05-20', '2023-01-05']
})

# --- DATASET 2: TRANSACTIONS (One-to-Many & Duplicates) ---
transactions = pd.DataFrame({
    'Trans_ID': ['T1', 'T2', 'T2', 'T3', 'T4', 'T5'], # Duplicate T2
    'Cust_ID': [101, 102, 102, 101, 105, 999], # 999 is a ghost customer (not in Cust table)
    'Amount': [500, 1200, 1200, 300, 450, 2000],
    'Trans_Date': ['2023-01-05', '2023-01-10', '2023-01-10', '2022-12-31', '2023-02-01', '2023-06-01'] 
    # Logic Error: T3 (2022) happened BEFORE the customer joined (2023)
})

print(" Customers Table ---")
print(customers.head())
print("\n Transactions Table ")
print(transactions.head())

import json

# --- TASK 1: JSON PARSING ---
# We define a helper function to safely parse the string
def extract_city(json_str):
    try:
        data = json.loads(json_str) # Convert string to Dictionary
        return data.get('City')
    except:
        return np.nan

# Apply the function
customers['City_Clean'] = customers['Address_JSON'].apply(extract_city)
# Standardize
customers['City_Clean'] = customers['City_Clean'].str.title().fillna('Unknown')

customers['Phone'] = customers['Contact_Info'].str.replace('-', '').str.extract(r'(\d{10})')

trans_clean = transactions.drop_duplicates()

cust_spend = trans_clean.groupby('Cust_ID')['Amount'].sum().reset_index()
cust_spend.rename(columns={'Amount': 'Total_Spend'}, inplace=True)

# Merge
df_final = pd.merge(customers, cust_spend, on='Cust_ID', how='left')

#  IMPUTATION ---
df_final['Total_Spend'] = df_final['Total_Spend'].fillna(0)

check_dates = pd.merge(df_final, trans_clean[['Cust_ID', 'Trans_Date']], on='Cust_ID', how='inner')

# Convert to datetime
check_dates['Join_Date'] = pd.to_datetime(check_dates['Join_Date'])
check_dates['Trans_Date'] = pd.to_datetime(check_dates['Trans_Date'])

# Create a boolean flag
check_dates['Suspicious_Flag'] = check_dates['Trans_Date'] < check_dates['Join_Date']

print("Final Processed Data ")
print(df_final[['Cust_ID', 'City_Clean', 'Phone', 'Total_Spend']])

print("\n Suspicious Transactions (Time Travel) ---")
print(check_dates[check_dates['Suspicious_Flag'] == True][['Cust_ID', 'Join_Date', 'Trans_Date']])