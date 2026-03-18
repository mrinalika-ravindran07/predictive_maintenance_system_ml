import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP: Advanced Data Generation ---
np.random.seed(99)
# Normal transactions: Log-normal distribution (common in finance for wealth/spending)
data = np.random.lognormal(mean=3, sigma=0.5, size=1000) 
# Inject Extreme Anomalies (Whales & Errors)
anomalies = [5000, 12000, -500, 0.01] 
final_data = np.concatenate([data, anomalies])

df = pd.DataFrame(final_data, columns=['Amount'])

# 2. THE ADVANCED ANALYTICAL ENGINE ---
def generate_financial_report(df, col):
    # A. Calculate "Robust" Stats vs "Standard" Stats
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Defining fences
    lower_fence = Q1 - (1.5 * IQR)
    upper_fence = Q3 + (1.5 * IQR)
    
    # Identifying the specific rows
    outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
    clean_data = df[(df[col] >= lower_fence) & (df[col] <= upper_fence)]
    
    # B. Create a Comparison Table
    report = pd.DataFrame({
        'Metric': ['Count', 'Mean (Avg)', 'Median (Typical)', 'Std Dev', 'Skewness'],
        'All Data (Risky)': [
            df[col].count(), df[col].mean(), df[col].median(), df[col].std(), df[col].skew()
        ],
        'Clean Data (Safe)': [
            clean_data[col].count(), clean_data[col].mean(), clean_data[col].median(), 
            clean_data[col].std(), clean_data[col].skew()
        ]
    })
    
    # Formatting for currency and readability
    print("\n --- FINANCIAL INTEGRITY REPORT --- 📊")
    print(report.round(2).to_string(index=False))
    
    print(f"\n OUTLIER DETECTION SUMMARY:")
    print(f"   > IQR Bounds: ${lower_fence:.2f} to ${upper_fence:.2f}")
    print(f"   > Total Outliers Found: {len(outliers)}")
    print(f"   > Impact on Average: Removing outliers drops Mean from ${df[col].mean():.2f} to ${clean_data[col].mean():.2f}")

    return df, outliers

# --- 3. ADVANCED VISUALIZATION ---
df, outliers = generate_financial_report(df, 'Amount')

plt.figure(figsize=(12, 6))
# Using a Violin Plot: Shows probability density (fat parts) vs Box Plot (summary)
sns.violinplot(x=df['Amount'], inner='quartile', color='lightgray')
# Overlaying the actual outlier points in red
sns.stripplot(x=outliers['Amount'], color='red', size=7, label='Detected Outliers')

plt.title('Transaction Distribution: Violin Density + IQR Outliers', fontsize=14)
plt.xlabel('Transaction Value ($)')
plt.xscale('log') # Log scale helps visualize massive differences in wealth
plt.legend()
plt.show()