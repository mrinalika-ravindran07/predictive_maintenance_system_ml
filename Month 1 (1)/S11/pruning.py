#This program uses the Iris dataset to show the difference between a "wild," unpruned tree and a controlled, pruned tree


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load Data
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = list(data.target_names)

# Split data (important to show generalization)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# MODEL 1:UNPRUNED (Overfitting)
# No max_depth limit means it will keep splitting until leaves are pure
tree_unpruned = DecisionTreeClassifier(random_state=42)
tree_unpruned.fit(X_train, y_train)

# MODEL 2: PRUNED (Generalizing)
# max_depth=3 stops the tree from growing too complex
tree_pruned = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=42)
tree_pruned.fit(X_train, y_train)

#VISUALIZATION 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

# Plot Unpruned
plot_tree(tree_unpruned, feature_names=feature_names, class_names=class_names, 
          filled=True, ax=axes[0], rounded=True)
axes[0].set_title(f"Unpruned Tree (Depth: {tree_unpruned.get_depth()})\nNotice the complexity", fontsize=14)

# Plot Pruned
plot_tree(tree_pruned, feature_names=feature_names, class_names=class_names, 
          filled=True, ax=axes[1], rounded=True)
axes[1].set_title(f"Pruned Tree (Depth: {tree_pruned.get_depth()})\nCleaner rules", fontsize=14)

plt.show()