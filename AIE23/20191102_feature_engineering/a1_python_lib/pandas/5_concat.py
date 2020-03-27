#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import pandas as pd
import numpy as np

# concatenating
# ignore index
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
print(df1)
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
print(df2)
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
print(df3)
print('\n----------------concat-axis=0-------------------')
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print("(1) test")
print(res)
print('\n----------------concat-axis=1-------------------')
res = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
print("(2) test")
print(res)