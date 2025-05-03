def is_perfect(number):
    if number <= 0:
        print("Error")
        return ""  # מחזיר מחרוזת ריקה כדי שלא יודפס None

    divisors_sum = sum(i for i in range(1, number) if number % i == 0)
    return divisors_sum == number


# בדיקות לדוגמה
#print(is_perfect(6))   # True - כי 1+2+3=6
#print(is_perfect(28))  # True - כי 1+2+4+7+14=28
#print(is_perfect(5))   # False - כי מחלקיו הם רק 1
print(is_perfect(-5))  # Error - כי המספר אינו חיובי