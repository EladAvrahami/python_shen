import numpy as np

a = np.array([200], dtype='uint8')
print(a + a)  # לא 400 אלא 144 (גלישה)

a = np.array([200], dtype='uint16')
print(a + a)  # 400
