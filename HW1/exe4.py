grade=int(input('Hello, please enter a grade'))
if grade<0 or grade>100:
    print('Wrong grade')
else:
    if grade<59:
        print('Failed')
    if grade>=59 and grade<=89:
        print('Passed')
    if grade>=90:
        print('Amazing')

