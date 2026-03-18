import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# 2. Set a professional theme globally
sns.set_theme(style="darkgrid") 

# 3. Create a scatter plot with semantic mapping
# hue: colors points by category
# size: scales points by value
sns.scatterplot(
    data=tips, 
    x="total_bill", 
    y="tip", 
    hue="time", 
    size="size"
)

plt.title("Bill vs Tip Analysis")
plt.show()