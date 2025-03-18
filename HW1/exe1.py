a=int(input("Hello, please enter a"))
b=int(input("Please enter b"))

remainder=0

if a%b==0:
    print("Match")
else:
    remainder=(a//b) #מחלק את האופרנד הראשון בשני עם עיגול התוצאה כלפי מטה.

    print(remainder)