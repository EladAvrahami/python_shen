import numpy as np


def quadratic_roots(a, b, c):
    coefficients = [a, b, c]
    roots = np.roots(coefficients)

    print("השורשים של המשוואה הם:", roots)

    # # חשב את הדלתא (delta)
    # delta = b**2 - 4*a*c
    #
    # # בדוק אם הדלתא שלילית, אפסית או חיובית
    # if delta < 0:
    #     print("למשוואה אין פתרונות ממשיים.")
    # elif delta == 0:
    #     root = -b / (2 * a)
    #     print(f"יש פתרון אחד: {root}")
    # else:
    #     root1 = (-b + math.sqrt(delta)) / (2 * a)
    #     root2 = (-b - math.sqrt(delta)) / (2 * a)
    #     print(f"הפתרונות הם: {root1} ו-{root2}")

# בקשת קלט מהמשתמש
# a = float(input("הכנס את המקדם a: "))
# b = float(input("הכנס את המקדם b: "))
# c = float(input("הכנס את המקדם c: "))

a,b,c =input('enter 3 int separated by space (they must be positive) :').split(' ')
a =float(a)
b=float(b)
c=float(c)

quadratic_roots(a, b, c)