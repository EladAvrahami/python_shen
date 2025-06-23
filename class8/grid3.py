import matplotlib.pyplot as plt
import numpy as np

rows, cols = 10, 10
grid = np.zeros((rows, cols))

fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', interpolation='none', extent=[0, cols, 0, rows])

# ציור קווי רשת
for x in range(cols + 1):
    ax.axvline(x, color='black', linewidth=0.5)
for y in range(rows + 1):
    ax.axhline(y, color='black', linewidth=0.5)

ax.set_xticks([])
ax.set_yticks([])
plt.show()
