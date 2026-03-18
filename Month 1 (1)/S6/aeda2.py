import pandas as pd
import seaborn as sns

from ydata_profiling import ProfileReport

# Load dataset
df = sns.load_dataset('titanic')

profile = ProfileReport(df, title="Titanic Dataset - Advanced EDA Report", explorative=True)

profile.to_file("titanic_eda_report.html")

print("EDA Report generated successfully as 'titanic_eda_report.html'")