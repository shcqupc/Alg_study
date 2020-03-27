# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
from sklearn.preprocessing import Imputer

import pandas as pd
data_train = pd.read_csv("a8_titanic/data/train.csv")

print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
# 问题1 空值填充
print(data_train[0:10])
# use most frequent, it is general for discrete and continuos data
df = data_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
print(df[0:10])

