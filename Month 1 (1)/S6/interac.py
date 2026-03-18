import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = sns.load_dataset('tips')


plt.figure(figsize=(10, 6))
sns.pointplot(x='day', y='total_bill', hue='smoker', data=df, 
              capsize=.1, linestyles=["-", "--"], markers=["o", "x"])
plt.title('Feature Interaction: Day & Smoker Status Effect on Bill')
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Map colors to a fourth dimension (e.g., sex)
colors = df['sex'].map({'Male': 'blue', 'Female': 'red'})

ax.scatter(df['total_bill'], df['tip'], df['size'], c=colors, alpha=0.6)

ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
ax.set_zlabel('Size')
plt.title('3D Feature Interaction: Bill vs Tip vs Size')
plt.show()