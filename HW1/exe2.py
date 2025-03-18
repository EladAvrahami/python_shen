a=int(input("Hello, Please enter a positive number"))
temp=0
sum=0
#print(15%2)
#print(15//10)
#print(1%2)
if a<=0:
    print('Negative number')
else:
    while a>0:
        temp=a%10
        if temp%2==1:
            sum+=temp
            a=a//10
        else:
            a=a//10
    print('the sum of the odd digits is:',sum)


