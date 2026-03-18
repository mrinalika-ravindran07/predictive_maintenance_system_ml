import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. Create a small dummy dataset
# Points: [x, y]
X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 7], [7, 8], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1]) # Class 0 (Blue), Class 1 (Red)

# 2. Define a new "Query" point
query_point = np.array([[4, 5]])

# 3. Visualization function
def plot_knn(k, metric_name='euclidean'):
    # Train KNN
    # p=1 is Manhattan, p=2 is Euclidean
    p_val = 1 if metric_name == 'manhattan' else 2
    
    knn = KNeighborsClassifier(n_neighbors=k, p=p_val)
    knn.fit(X_train, y_train)
    prediction = knn.predict(query_point)
    
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], 
                color='blue', label='Class 0', s=100)
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], 
                color='red', label='Class 1', s=100)
    
    # Plot Query Point
    plt.scatter(query_point[:,0], query_point[:,1], 
                color='green', label='Query Point', marker='*', s=200)
    
    plt.title(f"KNN (k={k}, Metric={metric_name})\nPredicted Class: {prediction[0]}")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()
    

    print(f"--- Distances from Query Point {query_point[0]} ({metric_name}) ---")
    for i, point in enumerate(X_train):
        if metric_name == 'euclidean':
            dist = np.sqrt(np.sum((point - query_point)**2))
        else: # Manhattan
            dist = np.sum(np.abs(point - query_point))
        print(f"Point {point}: Distance = {dist:.2f}")

plot_knn(k=4, metric_name='euclidean')