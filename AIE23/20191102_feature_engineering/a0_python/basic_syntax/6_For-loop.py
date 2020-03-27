listT1=['apple','banana','cake','dog']
# previous: for (int i; i < 10; i++)
#ordinary for loop
for x in listT1:
	print(x)
for y in range(1,5):
	print(y)
for z in range(1,11,2):
	listT1.append(z**2)

print(listT1)

#one-line for loop command
listT2=[w**2 for w in range(1,5)]
print(listT2)

#extract portions
print(listT2[:])
print(listT2[:3])
print(listT2[1:])
print(listT2[1:3])
print(listT2[-3:])