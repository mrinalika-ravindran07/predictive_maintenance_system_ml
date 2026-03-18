import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport


df_full = sns.load_dataset('diamonds')
df = df_full.sample(n=2000, random_state=42)  # Sampling 2000 rows for cleaner plots

print(f"Dataset Loaded. Shape: {df.shape}")
print("Variables: Carat, Cut, Color, Clarity, Depth, Table, Price, x, y, z")

#(Heatmaps & Pairplots)


def multivariate_analysis():
    print("\n--- Running Module 1: Multivariate Analysis ---")
    
    # 1. Advanced Correlation Heatmap (Focusing on Numeric relationships)
    # We drop categorical columns for the matrix calculation
    numeric_cols = df.select_dtypes(include=np.number)
    
    plt.figure(figsize=(12, 8))
    # Using the 'mask' argument to hide the upper triangle (it's symmetrical and redundant)
    mask = np.triu(np.ones_like(numeric_cols.corr(), dtype=bool))
    
    sns.heatmap(numeric_cols.corr(), mask=mask, annot=True, fmt=".2f", 
                cmap='BrBG', vmin=-1, vmax=1, center=0, linewidths=0.5)
    plt.title('Correlation Matrix (Lower Triangle Only)')
    plt.show()

    # 2. Multivariate Pairplot
    # We visualize Price vs Carat vs Depth, colored by 'Cut' quality
    # This analyzes 4 variables: Price(y), Carat(x), Depth(diagonal), Cut(color)
    cols_to_plot = ['price', 'carat', 'depth', 'table']
    
    g = sns.pairplot(df, vars=cols_to_plot, hue='cut', palette='Spectral', 
                     plot_kws={'alpha': 0.6, 's': 30}, diag_kind='kde', corner=True)
    g.fig.suptitle("Pairplot: Relationships by Cut Quality", y=1.02)
    plt.show()


#  Visualizing 4+ Variable Interactions


def complex_interaction_visualization():
    print("\n--- Running Module 2: 4+ Variable Interaction ---")
    
    # Goal: Visualize relationships between Price, Carat, Cut, and Clarity simultaneously.
    # Variable 1 (X-axis): Carat
    # Variable 2 (Y-axis): Price
    # Variable 3 (Color/Hue): Cut (Ideal to Fair)
    # Variable 4 (Columns): Clarity (I1 to IF)
    
    # We use a FacetGrid to separate the 4th variable into different mini-charts
    g = sns.FacetGrid(df, col="clarity", col_wrap=4, hue="cut", palette="turbo", height=3.5)
    
    # Map the scatter plot onto the grid
    g.map(sns.scatterplot, "carat", "price", alpha=0.7, edgecolor=None)
    g.add_legend(title="Cut Quality")
    
    g.fig.suptitle("4D Analysis: Price vs Carat | Split by Clarity (Grid) & Cut (Color)", y=1.02)
    plt.show()
    
    # Alternative: Bubble Plot (5 Variables)
    # X=Carat, Y=Price, Color=Cut, Size=Depth, Shape=Clarity (Too messy usually, so we limit to 4)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x="carat", y="price", 
        hue="cut",              # Variable 3: Color
        size="depth",           # Variable 4: Dot Size
        sizes=(20, 200),        # Range of dot sizes
        palette="viridis", 
        alpha=0.6
    )
    plt.title("Bubble Plot: Price vs Carat (Color=Cut, Size=Depth)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()

# MODULE 3: Automated EDA (Pandas Profiling)

def automated_eda():
    print("\n--- Running Module 3: Automated EDA Report ---")
    
    # We use the full dataset here for accurate statistics
    # 'minimal=True' speeds up computation for large datasets by skipping expensive computations
    profile = ProfileReport(df_full,
                        title="Diamonds Dataset Advanced Audit")
    
    # Save the report
    output_file = "diamonds_advanced_audit.html"
    profile.to_file(output_file)
    print(f"Report generated: {output_file}")
# MAIN EXECUTION
if __name__ == "__main__":

    multivariate_analysis()
    complex_interaction_visualization()
    automated_eda()
    
   