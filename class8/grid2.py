#color grid
import matplotlib.pyplot as plt
import numpy as np

# יצירת מטריצה בגודל 10x10 עם ערכים אקראיים
grid = np.random.rand(10, 10)

# ציור הרשת עם צבעים
plt.imshow(grid, cmap='viridis', interpolation='none')
plt.colorbar()  # סרגל צבע
plt.grid(True)

plt.show()
