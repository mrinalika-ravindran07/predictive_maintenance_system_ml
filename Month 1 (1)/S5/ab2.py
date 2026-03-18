import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
days = 30
daily_visitors = 500


p_A = 0.10  # Control: 10% conversion
p_B = 0.12  # Variant: 12% conversion (Winner)

data = []
for day in range(1, days + 1):
    # Simulate Group A
    conv_A = np.random.binomial(n=daily_visitors, p=p_A)
    data.append({'Day': day, 'Group': 'A', 'Visitors': daily_visitors, 'Conversions': conv_A})
    
    # Simulate Group B
    conv_B = np.random.binomial(n=daily_visitors, p=p_B)
    data.append({'Day': day, 'Group': 'B', 'Visitors': daily_visitors, 'Conversions': conv_B})

df = pd.DataFrame(data)

# Calculate Cumulative Stats for Trend Lines
df['Cumulative_Visitors'] = df.groupby('Group')['Visitors'].cumsum()
df['Cumulative_Conversions'] = df.groupby('Group')['Conversions'].cumsum()
df['Cumulative_Rate'] = df['Cumulative_Conversions'] / df['Cumulative_Visitors']


plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.lineplot(data=df, x='Day', y='Cumulative_Rate', hue='Group', palette='Set1', linewidth=2.5)
plt.title('Stability Check: Cumulative Conversion Rate')
plt.ylabel('Conversion Rate')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)

# Get totals
total_A = df[df['Group']=='A'].sum()
total_B = df[df['Group']=='B'].sum()

x = np.linspace(0.08, 0.15, 1000)

y_A = stats.beta.pdf(x, total_A['Conversions'] + 1, total_A['Visitors'] - total_A['Conversions'] + 1)
y_B = stats.beta.pdf(x, total_B['Conversions'] + 1, total_B['Visitors'] - total_B['Conversions'] + 1)

plt.plot(x, y_A, label='Group A (Control)', color='red', fillstyle='full')
plt.fill_between(x, y_A, alpha=0.2, color='red')
plt.plot(x, y_B, label='Group B (Variant)', color='blue')
plt.fill_between(x, y_B, alpha=0.2, color='blue')

plt.title('Uncertainty Analysis (Beta Distribution)')
plt.xlabel('True Conversion Rate Probability')
plt.yticks([]) # Hide y-axis as it's relative density
plt.legend()



plt.subplot(1, 3, 3)
sns.barplot(x='Group', y='Conversions', data=df, estimator=sum, palette='Set1', alpha=0.6)

plt.cla() # Clear axis
sns.barplot(x='Group', y='Conversions', data=df, errorbar=('ci', 95), palette='Set1')
plt.title('Avg Daily Conversions (with 95% CI)')

plt.tight_layout()
plt.show()


print(f"Group A Final Rate: {total_A['Conversions']/total_A['Visitors']:.2%}")
print(f"Group B Final Rate: {total_B['Conversions']/total_B['Visitors']:.2%}")