import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 100)
# print(x)
y1 = 6*x - 3.5
y2 = 14*np.cos(x)
plt.plot( x, y1, x, y2)
plt.xlabel('time(s)')
plt.ylabel('V(m/s)')
plt.title('title of graph')
plt.grid()
plt.show()