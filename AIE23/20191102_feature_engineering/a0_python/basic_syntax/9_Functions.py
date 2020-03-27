#Default value
def sayhello(user_name='Default name'):  
	return ("Hello! "+user_name)

def sayhello2(a):  
	return ("Hello! "+a)
	#return

print(sayhello())
print(sayhello('Daniel'))
#print(sayhello(a='Daniel'))

test = sayhello()
print(test)

#Multiple input
def team(a,*b):  
	print(a+" : "+str(b))

team("Alpha team","Carol")
team("Beta team","Hank","Tim","Daniel")
