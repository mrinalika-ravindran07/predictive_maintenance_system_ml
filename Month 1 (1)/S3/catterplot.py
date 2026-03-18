#Goal: Combining a scatter plot with histograms on the margins to see both the relationship and the individual distributions


import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# JointGrid gives you full control over the center plot and marginal plots
g = sns.JointGrid(data=iris, x="sepal_length", y="sepal_width", space=0)

# Define what goes in the center (KDE = Kernel Density Estimate / Contour plot)
g.plot_joint(sns.kdeplot, fill=True, cmap="rocket", thresh=0, levels=10)

# Define what goes on the sides (Histograms)
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)

plt.show()