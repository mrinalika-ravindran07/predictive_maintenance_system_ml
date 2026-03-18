#Goal: Creating "Small Multiples" (many small plots sharing axes) to compare subsets of data
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# lmplot = Linear Model Plot
# This draws a Scatter Plot + Linear Regression Line + Confidence Interval
# col="time" creates separate plots for Lunch vs Dinner
# row="sex" creates separate rows for Male vs Female
g = sns.lmplot(
    data=tips, 
    x="total_bill", 
    y="tip", 
    col="time", 
    row="sex", 
    hue="smoker",   # Different colors for smokers
    height=3,       # Height of each subplot
    aspect=1.2      # Width ratio
)

# Customizing the FacetGrid after creation
g.fig.suptitle("Complex Multivariate Regression Analysis", y=1.03)
g.set_axis_labels("Total Bill ($)", "Tip ($)")

plt.show()