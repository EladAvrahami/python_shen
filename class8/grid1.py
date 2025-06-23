import matplotlib.pyplot as plt

# יצירת פיגורה
fig, ax = plt.subplots()

# יצירת גריד (רשת) על הצירים
ax.set_xticks(range(11))  # קווים אנכיים כל 1 יחידה (0 עד 10)
ax.set_yticks(range(11))  # קווים אופקיים כל 1 יחידה (0 עד 10)
ax.grid(True)             # הפעלת הרשת

# הגדרת גבולות הגרף
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# הצגת הגרף
plt.show()
