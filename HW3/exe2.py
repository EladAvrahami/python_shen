import numpy as np

def makeList2RequierArr(lst):
    arr1 = np.array(lst)
    arr1[arr1 < 0] = -1  # תנאי מקוצר להמרת ערכים שליליים
    return arr1

my_list = range(10, -11, -2)
print("new arr is: " + " ".join(map(str, makeList2RequierArr(my_list))))

list_of_lists = [[1,2,3], [3,2,1], [4,5,6]]
matrix = np.array(list_of_lists)

print("Row 2 in matrix: " + str(matrix[1]))

# Broadcasting
a = np.array([2])

# יצירת הפלט בפורמט הרצוי עם פסיקים מופרדים בתוך הסוגריים
print("Broadcasting:", "[[" + "], [".join([",".join(map(str, row)) for row in matrix * a]) + "]]")