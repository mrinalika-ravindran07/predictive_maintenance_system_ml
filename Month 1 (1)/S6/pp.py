import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
df = sns.load_dataset('penguins')

# Drop missing values for clean plotting
df = df.dropna()

print("Dataset Preview:")
print(df.head())

sns.pairplot(df, hue='species', palette='husl')

plt.title("Multivariate Analysis: Pairplot of Penguin Data", y=1.02)
plt.show()