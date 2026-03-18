#Website Conversion Optimization. Goal: Demonstrate the "Peeking Problem." If you check your A/B test every day and stop when it's significant, your False Positive rate explodes.
import numpy as np
from scipy import stats

def run_simulation(n_simulations=1000):
    false_positives_peeking = 0
    false_positives_proper = 0
    
    print(f"Simulating {n_simulations} A/A tests (Where NO difference exists)...")
    
    for _ in range(n_simulations):
        # Generate data where Group A and Group B are IDENTICAL
        # Any 'significance' found here is a False Positive (Error)
        group_a = np.random.normal(100, 10, 1000)
        group_b = np.random.normal(100, 10, 1000)
        
    
        # We check the p-value every 100 samples. If it's ever < 0.05, we stop and claim victory.
        peeking_significance_found = False
        for i in range(100, 1001, 100):
            stat, p = stats.ttest_ind(group_a[:i], group_b[:i])
            if p < 0.05:
                peeking_significance_found = True
                break # We stopped early!
        
        if peeking_significance_found:
            false_positives_peeking += 1
            
        stat, p_final = stats.ttest_ind(group_a, group_b)
        if p_final < 0.05:
            false_positives_proper += 1

    print("\n--- RESULTS ---")
    print(f"False Positive Rate (Proper Method): {false_positives_proper/n_simulations:.1%} (Should be ~5%)")
    print(f"False Positive Rate (Peeking Method): {false_positives_peeking/n_simulations:.1%} (DANGEROUSLY HIGH!)")

run_simulation()