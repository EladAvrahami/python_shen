import math

# Get input from the user
num1 = int(input("Please enter a number: "))
num2 = int(input("Please enter a number: "))

# Validate the input
if num1 <= 0 or num2 <= 0 or num1 > num2:
    print("Wrong input")
else:
    # Calculate the sum of factorials in the range
    factorial_sum = sum(math.factorial(n) for n in range(num1, num2 + 1))
    print(factorial_sum)