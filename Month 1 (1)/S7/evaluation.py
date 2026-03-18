from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test):
    """Predicts and prints the error metrics."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Performance (MSE): {mse:.4f}")
    return mse