# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 
import numpy as np
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
print(__doc__)
# https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
# imputer 适合给连续数据用

import pandas as pd
data_train = pd.read_csv("../data/train.csv")
# use most frequent, it is general for discrete and continuos data
df = data_train.apply(lambda x:x.fillna(x.value_counts().index[0]))

print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
n_samples = 10
a = df["Fare"]
print(a)
# say you want to split at 1 and 3
#boundaries = [1, 3]
# add min and max values of your data
# boundaries = sorted({a.min(), a.max() + 1} | set(boundaries))
boundaries = np.linspace(a.min(), a.max() + 1, 9)
print('boundaries',boundaries)
# note should add labels=False
a_discretized_1 = pd.cut(a, bins=boundaries, labels=False) 
print(a[0:10], '\n')
print(a_discretized_1[0:10])


