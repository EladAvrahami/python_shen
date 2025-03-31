count = 1
while count <= 5:
    print("מספר:", count)
    count += 1
#שימוש ב- break ליציאה מהלולאה
num = 1
while num < 10:
    print(num)
    if num == 5:
        break  # יציאה מהלולאה כאשר num == 5
    num += 1

#שימוש ב- continue לדילוג על חזרה
num = 0
while num < 5:
    num += 1
    if num == 3:
        continue  # מדלג על ההדפסה כש-num שווה 3
    print(num)
