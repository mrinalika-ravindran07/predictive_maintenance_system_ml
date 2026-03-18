
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# GENERATE DATA
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2,   
    random_state=42
)
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# 3. TRAIN MODELS
# We must define and train 'rf_model' and 'xgb_model' here so the plotter can find them
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# 4. DEFINE PLOTTING FUNCTION
def plot_feature_importance(model, feature_names, model_name="Model"):
    # Extract importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort for better visualization
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, hue='Feature', palette='viridis', legend=False)
    
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# 5. EXECUTE PLOTS
# Now these variables exist, so the function will work
print("Generating plots...")
plot_feature_importance(rf_model, X_train.columns, "Random Forest")
plot_feature_importance(xgb_model, X_train.columns, "XGBoost")