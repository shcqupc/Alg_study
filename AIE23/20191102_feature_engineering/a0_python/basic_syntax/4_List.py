#Insert a element
listT1=['a','b','c','d','e','f']
list = []
dic = {"name":"harry"} #json
listT1.append('g')  # at the end of the list
print("1st")
print(listT1)

listT1=['a','b','c','d','e','f']
listT1.insert(0,'0')  # at the particular position
print("2nd")
print(listT1)

#Clear the position
listT1=['a','b','c','d','e','f']
print("3rd")
del listT1[0]
print(listT1)

#Remove the element
listT1=['a','b','c','d','e','f']
print("5th")
listT1.remove('b')
print(listT1)

#Length of the list
print("6th Length of the list:")
print(len(listT1))

#Range func.
print("7th Length of the list:")
listT3=list(range(2,10,2))
print(listT3)
