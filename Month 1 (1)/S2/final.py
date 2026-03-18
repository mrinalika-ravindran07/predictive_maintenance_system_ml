import pandas as pd
import numpy as np

# 1. Sales Data (Transaction Log)
sales = pd.DataFrame({
    'TransactionID': [101, 102, 103, 104],
    'Prod_ID': ['P1', 'P2', 'P1', 'P99'], 
    'Amount': [500, 700, 500, 200]
})

products = pd.DataFrame({
    'Prod_ID': ['P1', 'P2', 'P3'],
    'ProductName': ['Laptop', 'Mouse', 'Keyboard']
})

print("--- 1. Left Merge Sales with Products ---")

report = pd.merge(sales, products, on='Prod_ID', how='left')
print(report)


print("\n--- 2. Fill Missing Product Names ---")
report['ProductName'] = report['ProductName'].fillna('Unknown Product')

print("\n--- 3. Final Clean Report ---")
print(report)