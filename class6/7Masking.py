import numpy as np

a=np.array([10,20,30,40,50,12,15])
print(a<=20)

#לעבור על ערך התאים במערך וכל אחד גדול מ20 נשווה ל0
a[a>20]=0
print(a)