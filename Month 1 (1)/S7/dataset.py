import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes 

def get_data():
    """Fetches built-in data so it works offline."""
    print("Loading data (Offline mode)...")
    
   
    data = load_diabetes(as_frame=True)
    
    df = data.frame
    
    return df

def split_data(df):
    """Splits data into X (features) and y (target), then train/test."""
    # The target column name in load_diabetes is usually 'target'
    if 'target' in df.columns:
        X = df.drop(columns=['target'])
        y = df['target']
    else:
        # Fallback if the structure varies
        print("Columns available:", df.columns)
        raise ValueError("Could not find target column")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)