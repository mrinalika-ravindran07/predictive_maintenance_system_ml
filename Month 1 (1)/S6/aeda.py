import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

# Load dataset
df = sns.load_dataset('titanic')

# --- Automating EDA ---
# Create the profile object
profile = ProfileReport(df, title="Titanic Dataset EDA Report", explorative=True)


# Option 2: Save as an HTML file (
output_file = "titanic_eda_report.html"
profile.to_file(output_file)

print(f"Report saved as {output_file}. Open this file in your browser to view the analysis.")