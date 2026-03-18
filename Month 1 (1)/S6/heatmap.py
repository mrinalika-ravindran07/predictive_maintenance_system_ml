import seaborn as sns
import matplotlib.pyplot as plt

# Load sample dataset
df = sns.load_dataset('tips')


corr_matrix = df.corr(numeric_only=True)

# --- Creating a Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, 
            annot=True,      # Write the data value in each cell
            cmap='coolwarm', # Color scheme
            fmt=".2f",       # Format to 2 decimal places
            linewidths=0.5)

plt.title("Multivariate Analysis: Correlation Heatmap")
plt.show()