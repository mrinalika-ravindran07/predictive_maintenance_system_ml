import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

sns.lmplot(x="total_bill", 
           y="tip", 
           hue="smoker",  # Interaction variable
           data=df, 
           markers=["o", "x"], 
           palette="Set1")

plt.title("Feature Interaction: Bill vs Tip by Smoker Status")
plt.show()

# --- Feature Interaction: Boxplot with Subgroups ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", hue="sex", data=df, palette="Set3")
plt.title("Interaction: Day vs Bill by Gender")
plt.show()