sum=0

number=int(input("Hello, Please enter a positive number"))

if number <= 0:
     print("Negative number")

else:
     num=str(number)

     for digit in num:

         if int(digit)%2 != 0:

             sum += int(digit)
     print("the sum of the odd digits is:", sum)


