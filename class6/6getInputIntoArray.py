import  numpy as np



#first methode
my_arry=[]
n=int(input('size of array:'))
for i in range(n):
    my_arry.append(float(input('enter element:')))
my_arry=np.array(my_arry)
print(np.floor(my_arry))

#seconde methode
A=np.array(input('enter nums sparated by space:').split(" "),float)
print(np.floor(A))