import numpy as np

# Create a NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Create a mask: select elements greater than 25
mask = arr > 25
print("Mask:", mask)  # [False False  True  True  True]

# Use the mask to filter elements
filtered = arr[mask]
print("Filtered elements:", filtered)  # [30 40 50]

# Modify elements using the mask
arr[mask] = 0
print("Modified array:", arr)  # [10 20  0  0  0]
#how to print the mask
print("Mask:", mask)  # [False False  True  True  True]

