import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Scikit-learn modules
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Loading Iris Dataset...")
iris = datasets.load_iris()

# We will use only the first two features for easy 2D plotting: 
# Sepal Length & Sepal Width
X = iris.data[:, :2] 
y = iris.target

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (Important for KNN!)
# This scales data so one feature doesn't dominate the distance calculation.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Model A: Logistic Regression ---
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# --- Model B: K-Nearest Neighbors (KNN) ---
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n" + "="*40)
print("     PERFORMANCE REPORT")
print("="*40)

print(f"\n--- Logistic Regression Results ---")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names))

print(f"\n--- KNN (k={k}) Results ---")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

def plot_decision_boundary(X, y, classifier, title):
    # Create a mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict class for each point in the mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    
    # Plot also the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                          edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')

# Create the plots
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(X_train, y_train, log_reg, "Logistic Regression Boundaries")

plt.subplot(1, 2, 2)
plot_decision_boundary(X_train, y_train, knn, f"KNN (k={k}) Boundaries")

plt.tight_layout()
plt.show()