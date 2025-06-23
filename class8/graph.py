import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 100)
# print(x)
y = 5*np.sin(x)

plt.plot(x, y, c = 'r', lw = 3)
plt.grid()
plt.show()