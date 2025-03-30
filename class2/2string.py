print("it's okay") #use double quotes to print the '
print('it\'s ok') #use backslash so it read it like '
st="""this string 
is separated to
3 lines """

str="string"
print(str[2])

print(len(str)) #אורך סטרינג

# .split() = פיצול
# .join() #join strings together to 1
# str.capitalize(str) #אות ראשונה תמיד גדולה
# print(str)
# str.upper(str) #all letter capital
# print(str)
# str.lower(str)
# print(str)

s1 ='      kuku'
print(s1.strip()) #delete spaces
s2='aaaasssgaafbbbtt'
print(s2.index('f'))

if s2.find("g")==-1:
    print('not found')
else:
    print('the letter g find at index:',s2.find('g'))

str4='abcdefg'
print(str4[3:6])#print range
print(str4[0:3])#example 2
print(str4[::-1]) #print backwards
print(str4[3::-1]) #print backwards from specific index
print(str4[0:6:2])#print range jump 2 every print
