#טאפל הוא כמו רשימה, אבל קבוע – אי אפשר לשנות אותו אחרי יצירתו (immutable).

#craete tuple
tup = (5, 10, 15)
print(tup)  # (5, 10, 15)
print(tup[1])  # by index

#tup[0] = 7  # ❌ תגרום לשגיאה: 'tuple' object does not support item assignment

#connect 2 tuples
tup1 = (1, 2)
tup2 = (3, 4)
tup3 = tup1 + tup2
print(tup3)  # (1, 2, 3, 4)

#חיתוך (Slicing) כמו ברשימה
tup = (10, 20, 30, 40)
print(tup[1:3])  # (20, 30)

#השמה מרובה (Multiple Assignment)

# טאפל מאפשר לפרק ערכים למשתנים בבת אחת:

(a, b, c) = (5, 10, 15)
print("a =", a, "b =", b, "c =", c)  # a = 5 b = 10 c = 15

#החלפת ערכים בלי משתנה עזר
x = 1
y = 2
x, y = y, x
print(x, y)  # 2 1

#get 3 num inside tupple
tup = (input("1: "), input("2: "), input("3: "))
print(tup)





