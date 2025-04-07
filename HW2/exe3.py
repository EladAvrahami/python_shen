
def is_prime(number):
    """Check if a number is prime."""
    if number < 2:
        return False
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    return True

# Let the user enter two numbers
num1 = int(input("Please enter a number: "))
num2 = int(input("Please enter a number: "))

# Validate the input
if num1 <= 2 or num2 <= 2 or num1 > num2:
    print("Wrong input")
else:
    # Find and print prime numbers in the range
    prime_numbers = [n for n in range(num1, num2 + 1) if is_prime(n)]
    if prime_numbers:
        print(", ".join(map(str, prime_numbers)) + ",")
    else:
        print("No prime numbers found in the range.")




















# def is_prime(number):
#     """בודק אם המספר ראשוני"""
#     if number < 2:
#         return False
#     for i in range(2, int(number**0.5) + 1):
#         if number % i == 0:
#             return False
#     return True
#
# def get_positive_input():
#     """מקבל מספר שלם חיובי (גדול מ-2) מהמשתמש ובודק תקינות"""
#     try:
#         num = int(input('Please enter a number'))
#         if num > 2:
#             return num
#         else:
#             print("Wrong input")
#             return None
#     except ValueError:
#         print("Wrong input")
#         return None
#
# # קבלת שני מספרים חיוביים מהמשתמש
# num1 = get_positive_input()
# if num1 is None:
#     exit()
#
# num2 = get_positive_input()
# if num2 is None:
#     exit()
#
# # יצירת רשימה של מספרים ראשוניים בתחום
# prime_numbers = [n for n in range(num1, num2 + 1) if is_prime(n)]
#
# # הדפסת התוצאה
# if prime_numbers:
#     print(", ".join(map(str, prime_numbers)) + ",")
# else:
#     print("No prime numbers found in the range.")