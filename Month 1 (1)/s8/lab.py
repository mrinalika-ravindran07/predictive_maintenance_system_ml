import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load Data  ---
try:
    df = pd.read_csv('housing.csv')
    print(" Successfully loaded 'housing.csv'")
except FileNotFoundError:
    print(" Error: 'housing.csv' not found. Please make sure it is in the same folder as this script.")
    exit()



# Handle missing values (Raw CSV often has missing values in total_bedrooms)
df = df.dropna()


if 'total_rooms' in df.columns:
    df['AveRooms'] = df['total_rooms'] / df['households']
    df['AveBedrms'] = df['total_bedrooms'] / df['households']
    df['AveOccup'] = df['population'] / df['households']
    df['MedInc'] = df['median_income']
    df['HouseAge'] = df['housing_median_age']
    df['Price'] = df['median_house_value'] / 100000 # Scale price to match sklearn format ($100k units)
    
    
    drop_cols = ['total_rooms', 'total_bedrooms', 'households', 'median_income', 
                 'housing_median_age', 'median_house_value', 'ocean_proximity']
    # Only drop columns that actually exist
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

print(f"Processed Dataset Shape: {df.shape}")
print("-" * 30)

df['Rooms_per_Household'] = df['AveRooms'] 
df['Bedrooms_per_Room'] = df['AveBedrms'] / df['AveRooms']
df['Population_per_Household'] = df['Population'] / df['AveOccup'] if 'Population' in df.columns else df['population'] / df['AveOccup']

# Define X and y
# We drop the target 'Price' and intermediate calculation columns
X = df.drop(['Price', 'AveRooms', 'AveBedrms'], axis=1)
if 'Population' in df.columns: X = X.drop(['Population'], axis=1)
if 'population' in df.columns: X = X.drop(['population'], axis=1)

y = df['Price']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Building the Advanced Pipeline ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', Ridge())
])

# --- 5. Hyperparameter Tuning ---
param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 100.0]}

print("Tuning hyperparameters... (This may take a moment)")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Alpha Found: {grid_search.best_params_['regressor__alpha']}")

# --- 6. Evaluation ---
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: ${rmse*100000:,.2f} (Average error in dollars)")
print(f"R² Score: {r2:.4f} (Variance explained)")

# --- 7. Diagnostic Visualizations ---
plt.figure(figsize=(14, 6))

# Plot A: Actual vs Predicted
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("Actual vs. Predicted Prices")

# Plot B: Residuals Distribution
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Residuals (Error)")
plt.title("Distribution of Errors")

plt.tight_layout()
plt.show()