#Visualizing relationships between all numeric variables at once. This is crucial for Data Science Exploratory Data Analysis (EDA)
# import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
flights =sns.load_dataset("flights")

# Pivot the data to get a matrix form (Month vs Year)
# This format is required for heatmaps
flights_matrix = flights.pivot(index="month", columns="year", values="passengers")

plt.figure(figsize=(10, 8))

# Create Heatmap
# annot=True writes the numbers inside the boxes
# cmap="YlGnBu" sets the color map (Yellow-Green-Blue)
sns.heatmap(
    flights_matrix, 
    annot=True, 
    fmt="d",       # "d" means integer formatting (no decimals)
    cmap="YlGnBu", 
    linewidths=.5  # Adds white lines between cells
)

plt.title("Airline Passengers Heatmap (1949-1960)")
plt.show()