import numpy as np



#
print(np.zeros((2,3)))# מערך מלא אפסים
print(np.ones((3,3))) # מערך מלא באחדים
print('******************')
print(np.full((2,2), 7)) # מערך מלא בערך קבוע
print('******************')
print(np.eye(5))# מטריצת יחידה
print('******************')
print(np.random.rand(2,2))# מערך רנדומלי
print('******************')
print(np.random.randint(0, 10, (2, 3)))# מערך רנדומלי של מספרים שלמים
#This will create a 2x2 array with random integers between 0 (inclusive) and 10 (exclusive). Adjust the range (0, 10) as needed.

print('*********שימוש ב- arange ו- reshape********************')

arr=np.arange(36)
print(arr)
print('********** use resahpe the multiple of the 2/3D should be as the range num ratio:  ********')
arr=np.arange(36).reshape(3,4,3)
print(arr)
arr=np.arange(36).reshape(18,2)
print(arr)

print()
print( ' Transpose - היפוך צירים במערך')
arr2=np.arange(9).reshape(3,3)
print(arr2)
print(np.transpose(arr2))