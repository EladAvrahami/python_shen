import math as m

# קבלת מקדמי המשוואה
a, b, c = input("Enter the coefficients a, b, c separated by spaces: ").split(" ")
a = float(a)
b = float(b)
c = float(c)

# חישוב הדיסקרימיננטה
disc = b**2 - 4*a*c

# בדיקת סוג הדיסקרימיננטה
if disc < 0:
    print("No real roots.")
elif disc == 0:
    root = -b / (2 * a)
    print("There is one root:", root)
else: # disc > 0
    x1 = (-b + m.sqrt(disc)) / (2 * a)
    x2 = (-b - m.sqrt(disc)) / (2 * a)
    print("The roots are: {:.2f}, {:.2f}".format(x1, x2))