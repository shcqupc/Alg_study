#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
#Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
#python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，
#永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
"""
from __future__ import print_function
import pandas as pd
# Read from
data = pd.read_csv('student.csv')
print(data)

# Save to
data.to_pickle('student.pickle')

# Load pickle
import pprint, pickle
pkl_file = open('student.pickle', 'rb')
data1 = pickle.load(pkl_file)
pprint.pprint(data1)