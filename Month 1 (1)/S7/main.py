import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    # --- 1. Setup Directories ---
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created 'models' directory.")

    # --- 2. Load Data ---
    print("Loading Diabetes dataset...")
    # This loads the standard dataset (already scaled)
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # --- 3. Train Model ---
    print("Training Linear Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- 4. Evaluate (Optional) ---
    score = model.score(X_test, y_test)
    print(f"Model trained! R^2 Score: {score:.2f}")

    # --- 5. Save Model ---
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"Success! Model saved to {model_path}")

if __name__ == "__main__":
    main()