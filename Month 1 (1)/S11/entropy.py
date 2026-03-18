import numpy as np
import pandas as pd

def calculate_entropy(y):
    """Calculates entropy of a target array."""
    class_counts = np.bincount(y)
    probabilities = class_counts[class_counts > 0] / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_information_gain(parent, left_child, right_child):
    """Calculates IG given parent and split children."""
    w_left = len(left_child) / len(parent)
    w_right = len(right_child) / len(parent)
        
    entropy_parent = calculate_entropy(parent)
    entropy_children = (w_left * calculate_entropy(left_child)) + \
                       (w_right * calculate_entropy(right_child))
    
    gain = entropy_parent - entropy_children
    return entropy_parent, entropy_children, gain


parent_node = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
left_bad = np.array([0, 0, 1, 1])
right_bad = np.array([0, 0, 0, 1, 1, 1])

left_good = np.array([0, 0, 0, 0, 0])  # All No
right_good = np.array([1, 1, 1, 1, 1]) # All Yes

print("--- Scenario A: Bad Split ---")
e_p, e_c, ig = calculate_information_gain(parent_node, left_bad, right_bad)
print(f"Parent Entropy: {e_p:.3f}")
print(f"Information Gain: {ig:.3f} (Low Gain)\n")

print("--- Scenario B: Good Split ---")
e_p, e_c, ig = calculate_information_gain(parent_node, left_good, right_good)
print(f"Parent Entropy: {e_p:.3f}")
print(f"Information Gain: {ig:.3f} (Maximum Gain)")