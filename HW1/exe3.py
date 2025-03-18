a=int(input("Please enter your first number"))
b=int(input("Please enter a second bigger number"))
i=a
sum=0


if a>=b:
    print('Wrong Input')
else:
    while i<=b:
        sum+=+i
        i=i+1
    print('The sum is:',sum)
