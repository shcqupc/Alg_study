
Dc2='Resources/pi_digits.txt'

##Read entire file at once 
with open(Dc2) as FileTP2:  #Function 'with' indicates only when the file needed to be read that it will be opened. 
	contents = FileTP2.read() 
	print(contents.rstrip())  #With 'open' function, it will leave one blank line at the end when it finished reading the file 

#Read the user's inputs
# InputTest=input("text something")  #Install SublimeREPL so that u can run codes with input needs
# print(InputTest)