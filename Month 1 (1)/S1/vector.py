import numpy as np
import time

# Create a large array of 1 million numbers
data = np.arange(1_000_000)

# 1. Using a loop (Slow)
start = time.time()
loop_result = [x * 2 for x in data]
print(f"Loop time: {time.time() - start:.4f} seconds")

# 2. Using Vectorization (Fast)
start = time.time()
vector_result = data * 2  # No loop required!
print(f"Vectorized time: {time.time() - start:.4f} seconds")