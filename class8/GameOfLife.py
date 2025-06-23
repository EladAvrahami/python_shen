import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# יצירת לוח אקראי (1 = חי, 0 = מת)
def create_grid(rows, cols):
    return np.random.choice([0, 1], size=(rows, cols))

# חישוב הדור הבא
def update(frame_num, img, grid):
    new_grid = grid.copy()
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            # סכום השכנים החיים (כולל את התא עצמו - נוריד אותו)
            total = np.sum(grid[max(i-1,0):min(i+2,rows),
                                max(j-1,0):min(j+2,cols)]) - grid[i, j]

            # כללים:
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0  # מוות
            else:
                if total == 3:
                    new_grid[i, j] = 1  # לידה

    img.set_data(new_grid)
    grid[:] = new_grid
    return img,

# הגדרות ראשוניות
rows, cols = 50, 50
grid = create_grid(rows, cols)

fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='gray')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid), interval=200)
plt.title("Conway's Game of Life")
plt.show()
