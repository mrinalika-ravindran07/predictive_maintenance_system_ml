import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def visualize_p_value(observed_z_score):
    """
    Visualizes the P-value for a two-tailed test based on a Z-score.
    """
    # 1. Setup the Standard Normal Distribution (Null Hypothesis)
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Null Hypothesis Distribution', color='black')

    
    p_value = 2 * (1 - stats.norm.cdf(abs(observed_z_score)))

    # 3. Shade the Critical Regions (The P-Value area)
    # Right tail
    plt.fill_between(x, y, where=(x >= abs(observed_z_score)), color='red', alpha=0.5, label='P-Value Region')
    # Left tail
    plt.fill_between(x, y, where=(x <= -abs(observed_z_score)), color='red', alpha=0.5)

    # 4. Add visual markers
    plt.axvline(observed_z_score, color='blue', linestyle='--', label=f'Observed Z: {observed_z_score}')
    plt.title(f'Visualizing P-Value for Z-score = {observed_z_score}\nCalculated P-Value: {p_value:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return p_value

5
my_z_score = 2.0 
p = visualize_p_value(my_z_score)

if p < 0.05:
    print("Result: Statistically Significant (Reject Null)")
else:
    print("Result: Not Significant (Fail to Reject Null)")