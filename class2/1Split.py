x ,y =input('enter 2 int separated by # (they must be positive) :').split('#')
x =int(x)
y=int(y)
if x>0 and y>0:
    print(x+y,x-y,x*y)
else:
    print('err negative num')