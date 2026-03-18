import numpy as np
import time


raw_data = np.random.uniform(low=-600, high=120, size=10_000_000)

#APPROACH 1: TRADITIONAL LOOP-BASED TRANSFORMATION ---
def process_with_loops(data):
    result = []
    for temp in data:
        # Filter: Only keep readings above absolute zero (~ -459.67 F)
        if temp > -459.67:
            # Conversion formula: (F - 32) * 5/9
            celsius = (temp - 32) * (5/9)
            result.append(celsius)
    return result

print("Starting Loop-based processing...")
start_time = time.time()
loop_output = process_with_loops(raw_data)
loop_duration = time.time() - start_time
print(f"Loop Duration: {loop_duration:.4f} seconds")


#  APPROACH 2: OPTIMIZED NUMPY VECTORIZATION ---
def process_with_numpy(data):
    # Vectorized Filter: Create a boolean mask for valid temperatures
    mask = data > -459.67
    valid_data = data[mask]
    
    # Vectorized Transformation: Perform math on the entire array at once
    celsius_array = (valid_data - 32) * (5/9)
    return celsius_array

print("\nStarting NumPy Vectorized processing...")
start_time = time.time()
numpy_output = process_with_numpy(raw_data)
numpy_duration = time.time() - start_time
print(f"NumPy Duration: {numpy_duration:.4f} seconds")

#  RESULTS ANALYSIS
speedup = loop_duration / numpy_duration
print("-" * 30)
print(f"Speed Improvement: {speedup:.1f}x faster")
print(f"Data processed correctly? {len(loop_output) == len(numpy_output)}")