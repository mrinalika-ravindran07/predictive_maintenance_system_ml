import pandas as pd
from scipy import stats

def descriptive_audit(df, column):
    # Drop missing values to avoid errors in calculation
    col_data = df[column].dropna()

    # 1. Central Tendency & Spread
    mean = col_data.mean()
    median = col_data.median()
    std = col_data.std()
    
    # 2. Shape
    skew = col_data.skew()
    kurt = col_data.kurtosis()
    
    # 3. Logic Checks (The Audit part)
    warnings = []
    
    # Check for Skewness
    if abs(skew) > 1:
        warnings.append(f"Data is highly skewed ({skew:.2f}). Use Median, not Mean.")
        
    # Check for Outliers (Z-Score method)
    # Using scipy.stats.zscore
    z_scores = stats.zscore(col_data)
    outliers = col_data[abs(z_scores) > 3]
    if len(outliers) > 0:
        warnings.append(f"Detected {len(outliers)} extreme outliers (Z > 3).")
        

    cv = (std / mean) if mean != 0 else 0 
    if cv > 1:
        warnings.append(f" High Variance (CV: {cv:.2f}). Data is volatile.")

    # Report Formatting
    print(f"\n--- DATA AUDIT REPORT: {column.upper()} ---")
    print(f"Mean:           {mean:,.2f}")
    print(f"Median:         {median:,.2f}")
    print(f"Std Dev:        {std:,.2f}")
    print(f"Skewness:       {skew:.2f}")
    print(f"Kurtosis:       {kurt:.2f}")
    
    if warnings:
        print("\n--- WARNINGS & ACTION ITEMS ---")
        for w in warnings:
            print(w)
    else:
        print("\n Data looks normal and clean.")
    print("-" * 35)

# --- Test Case ---
salary_data = pd.DataFrame({'salary': [50000, 52000, 55000, 48000, 5000000]})
descriptive_audit(salary_data, 'salary')