# Goal: Create a list of squares for even numbers between 0 and 9
# Standard way
squares_loop = []
for x in range(10):
    if x % 2 == 0:
        squares_loop.append(x**2)

# List Comprehension way
squares_comp = [x**2 for x in range(10) if x % 2 == 0]

print(f"Loop version: {squares_loop}")
print(f"Comprehension version: {squares_comp}")