import numpy as np

def func1(lst):
    tmpLst = []
    for i in lst:
        if i%3 == 0:
         tmpLst.append(i//3)
        else:
            tmpLst.append(i)
    return tmpLst

def func2(lst):
    tmpLst = []
    for i in range(len(lst)):
        if i%3 == 0:
            tmpLst.append(i//3)
        else:
            tmpLst.append(i)
    return tmpLst

def func3(lst):
    tmpLst = []
    for i in range(len(lst)):
        if lst[i]%3 == 0:
            tmpLst.append(i//3)
        else:
            tmpLst.append(i)
    return tmpLst

def func4(lst):
    tmpLst = []
    for i in range(len(lst)):
        if lst[i]%3 == 0:
            tmpLst.append(lst[i]//3)
        else:
            tmpLst.append(i)
    return tmpLst


def func5(lst):
    tmpLst = []
    for i in range(len(lst)):
        if lst[i]%3 == 0:
            tmpLst.append(lst[i]//3)
        else:
            tmpLst.append(lst[i])
    return tmpLst


print("Func2: ", func2([1,2,3,4,5,6,7,8,9,10]))