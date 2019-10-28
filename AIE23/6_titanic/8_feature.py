# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
"""
# (4) 特征工程 - 特征抽取
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
"""
print(__doc__)


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("data/fix_data_tai2.csv")

df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#train_df.to_csv("processed_titanic.csv" , encoding = "utf-8")
df.to_csv("data/fix_data_tai3.csv")