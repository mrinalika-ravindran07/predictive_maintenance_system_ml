import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Create dummy dataset for Loan Approval
# 0 = Denied, 1 = Approved
data = {
    'Credit_Score': [600, 800, 750, 500, 550, 850, 700, 400, 650, 820],
    'Income_Level': [20, 80, 70, 15, 25, 90, 60, 10, 50, 85], # In thousands
    'Has_Debt':     [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],           # 1=Yes, 0=No
    'Loan_Approved':[0, 1, 1, 0, 0, 1, 1, 0, 0, 1]            # Target
}
df = pd.DataFrame(data)

# 2. Separate Features and Target
X = df[['Credit_Score', 'Income_Level', 'Has_Debt']]
y = df['Loan_Approved']

# 3. Train the Decision Tree
# We limit max_depth=3 to keep the plot readable
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X, y)

# 4. Visualize the Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=X.columns, 
          class_names=['Denied', 'Approved'], 
          filled=True, 
          rounded=True,
          fontsize=10)

plt.title("Loan Approval Decision Logic")
plt.show()