count = 0
sum = 0
avg = 0.0

num = int(input('Please enter a number'))
while num >= 0:
    count += 1
    sum += num
    num = int(input('Please enter a number'))

if count > 0:  # בדיקה אם הוזנו מספרים חיוביים
    avg = sum / count
    print("The length is: {}".format(count))
    print("The sum is: {}".format(sum))
    print("The average is: {:.1f}".format(avg))  # פורמט למספרים עשרוניים
else:
    print("The length is: {}".format(count))
    print("The sum is: {}".format(sum))
    print("The average is: {:.1f}".format(avg))
