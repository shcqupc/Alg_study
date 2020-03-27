# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
"""
数据类型与分布研究
对于列名查看如下：
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
其中有的类数据类型是object，也就是字符类型数据。
统计字符数量
"""
print(__doc__)

import numpy as np
import pandas as pd
data_train = pd.read_csv("data/train.csv")

data = data_train.values

# 观察所有数值型数据
for idx, itr in enumerate(data_train.dtypes):
    if itr == np.object:
        print("第%d列字符类个数："%idx, len(set(data[:, idx])))

print("""
很明显一些列中包含的字符类比较多，对于这些数据而言是不适合做分类算法的，因此需要将其剔除
在一些算法中是可以容忍类似数据的
""")





