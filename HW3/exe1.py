import numpy as np

def concat_lists(first_list, second_list):
    #conver lists to arrays:
    arr1=np.array(first_list)
    arr2=np.array(second_list)
    for cell in arr1:
        arr1[arr1 < 0] = 0 # תנאי מקוצר עובר על ירך תאים במערך ומשנה ל0 אם שלילי
    for cell in arr2:
        arr2[arr2 < 0] = 0
    concatenated_array=np.concatenate((arr1, arr2))
    return concatenated_array

# #testing
# list_a = [1, -2, 3, 4]
# list_b = [5, 6, 7, 8]
# concat_lists(list_a,list_b)
# print(concat_lists(list_a,list_b))



