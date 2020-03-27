#1st while loop
print("1st loop:")
num1=0
while num1<=50:
	print("\t"+str(num1))
	num1+=1
	if num1==5:
		print("break")
		break
	else:
		print("continue")
		continue
