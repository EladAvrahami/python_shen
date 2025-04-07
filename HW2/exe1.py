

#לולאה מקוננת פשוטה
num=int(input('Please enter a number'))
count = num
if 1 < num < 10:
    for square_y in range( num):
        for square_x in range(num-square_y):
            print(num-square_y, end='')#end='' used to print without printl in java
        print()
else:
    print('Wrong input')



