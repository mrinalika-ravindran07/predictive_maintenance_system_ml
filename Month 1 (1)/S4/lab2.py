import pandas as pd
import numpy as np
from typing import Tuple

class FinancialAuditor:
    """
    A production-ready class to ingest financial ledgers 
    and sanitize them using statistical thresholds.
    """
    
    def __init__(self, sensitivity: float = 1.5):
        """
        :param sensitivity: The IQR multiplier (Standard is 1.5, Aggressive is 3.0)
        """
        self.factor = sensitivity
        self.bounds = {}

    def fit_transform(self, df: pd.DataFrame, col_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates bounds and splits data into 'Clean' and 'Anomalies'.
        Returns: (clean_df, anomalies_df)
        """
        # 1. Calculation Phase
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (self.factor * IQR)
        upper_bound = Q3 + (self.factor * IQR)
        
        # Store logic for audit trails
        self.bounds = {
            "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "Lower": lower_bound, "Upper": upper_bound
        }

        # 2. Segmentation Phase
        mask_outlier = (df[col_name] < lower_bound) | (df[col_name] > upper_bound)
        
        anomalies_df = df[mask_outlier].copy()
        clean_df = df[~mask_outlier].copy()
        
        # Tagging the anomalies with a reason
        anomalies_df['Status'] = np.where(anomalies_df[col_name] > upper_bound, 
                                          'CRITICAL_HIGH', 'CRITICAL_LOW')
        
        return clean_df, anomalies_df

    def print_audit_log(self):
        """Prints the logic used for the last transformation."""
        print(f"\n[AUDIT LOG] Bounds set at: {self.bounds['Lower']:.2f} to {self.bounds['Upper']:.2f}")
# 1. Load Data
raw_data = pd.DataFrame({'txn_id': range(1, 11), 
                         'amount': [50, 55, 48, 52, 5000, 49, 10, -50, 60, 51]})

# 2. Initialize the Auditor (The "Tool")
auditor = FinancialAuditor(sensitivity=1.5)

# 3. Process the Data
print("Processing Transactions...")
valid_txns, flagged_txns = auditor.fit_transform(raw_data, 'amount')

# 4. Review Results
auditor.print_audit_log()

print("\n APPROVED TRANSACTIONS:")
print(valid_txns.to_string(index=False))

print("\n FLAGGED FOR REVIEW:")
# Using styling to highlight the dangerous column
print(flagged_txns.to_string(index=False))