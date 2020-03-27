#!/usr/bin/python
import numpy as np
import pandas as pd
from pandas import DataFrame
data = DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('wxyz'))
data['w']  #选择表格中的'w'列，使用类字典属性,返回的是Series类型
data.w    #选择表格中的'w'列，使用点属性,返回的是Series类型
data[['w']]  #选择表格中的'w'列，返回的是DataFrame类型
data[['w','z']]  #选择表格中的'w'、'z'列

print(data[0:2])  #返回第1行到第2行的所有行，前闭后开，包括前不包括后
print(data[1:2])  #返回第2行，从0计，返回的是单行，通过有前后值的索引形式 #如果采用data[1]则报错
print(type(data[0:2]))
print('\n------------------------------------')
print(data.columns)
print(data.info())