def findMax():
    lst=[1,5,10,20,60,30]
    max_val = lst[0]
    for i in lst:
        if i > max_val:
            max_val = i
    print("The Maximum value is:", max_val)

findMax()


#defult values function:
def greet(name="student"):
    print("Hello,", name)

greet() #print with the default value "student"
greet('david')

#פונקציה עם כמות פרמטרים לא ידועה

def unknonParamsVals(*params):
    for param in params:
        print(param)

unknonParamsVals('elad','roy',4,255,'roni')


# דוגמא החזרת שתי תוצאות
#פתרון משוואה ריבועית
import math
def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return "No real solutions"
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    return [x1, x2]
print(quadratic(1,-3,2))