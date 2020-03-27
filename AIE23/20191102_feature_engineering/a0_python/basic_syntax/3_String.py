string=" test "
number=1
# (1) regex, re
# (2) .split("")
#Change the Upper/Lower case
print("Change the Upper/Lower case")  
print(string.upper())
print(string.lower())
print("\n")

#Insert the Tab
print("Insert the Tab")
print("\t"+string)
print("\n")

#Clear the space
print("Clear the space")  
print(string.rstrip())
print(string.lstrip())
print(string.strip())
print("\n")

#Change the variable type
print("Change the variable type") 
print(str(number))

# import re
# "abcdefc" find "de"