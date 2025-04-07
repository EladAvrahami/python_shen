# while True:
#     s = input("Enter a string with letters only (no digits or special characters): ")
#     if s.isalpha():
#         print(f"The length of the string is: {len(s)}")
#         break
#     else:
#         print("Invalid input. Please try again.")

while not (s := input("Enter a string with letters only: ")).isalpha(): print("Invalid input. Try again.")
print(f"The length of the string is: {len(s)}")