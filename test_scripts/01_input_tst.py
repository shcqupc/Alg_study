# -*-conding:utf-8-*-
import os
import json

d1 = 'Resources/inputtest1.json'
d2 = 'Resources/inputtest2.txt'
d3 = 'Resources/outputtest1.json'
d4 = 'Resources/outputtest2.txt'
with open(d1) as f1:
    print('type(f1)', type(f1))
    numbers = json.load(f1)
    print(numbers)

with open(d2) as f2:
    content = f2.read()
    print(f2.read())

with open(d3, 'a+') as f3:
    json.dump(numbers, f3)
    f3.close()

with open(d4, 'a+') as f4:
    f4.write(content)
    f4.close()

#os.remove(d1)