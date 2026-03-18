from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):
    """Initializes and trains the model."""
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath="models/model.pkl"):
    """Saves the trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")