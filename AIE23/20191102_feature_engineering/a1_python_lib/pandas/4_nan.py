#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
df.iloc[5,0] = np.nan
df.iloc[4,3] = np.nan

print(df)
print('\n-----------------dropna-------------------')
# print(df.dropna())   # dropna默认丢弃任何含有缺失的行：
# print(df)
print('\n-----------------fillna-------------------')
print(df['A'].fillna(value=-1))
print(pd.isnull(df))
print(pd.notnull(df))
print('\n----------------fillna mean--------------------')
# print(df.A.fillna(value=8))
print(df.columns[df.isnull().sum(axis=0)>0])
# print(df.columns[df.isnull()])
for col in list(df.columns[df.isnull().sum(axis=0)>0]):
    df[col].fillna(value=df[col].mean(), inplace=True)
print(df)