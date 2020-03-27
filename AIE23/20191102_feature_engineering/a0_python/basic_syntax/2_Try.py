#!/usr/bin/python
# -*- coding: UTF-8 -*-
# try:
# <语句>        #运行别的代码
# except <名字>：
# <语句>        #如果在try部份引发了'name'异常
# except <名字>，<数据>:
# <语句>        #如果引发了'name'异常，获得附加的数据
# else:
# <语句>        #如果没有异常发生
try:
    file = open('a0_python/12_Class.py', 'r+')
except Exception as e:
    print('there is no file named as eeeee')
    response = input('do you want to create a new file')
    if response =='y':
        file = open('eeee','w')
    else:
        pass
else:
    file.write('ssss')
file.close()


