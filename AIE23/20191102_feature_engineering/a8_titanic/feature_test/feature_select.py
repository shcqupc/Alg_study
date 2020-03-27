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
data_train = pd.read_csv("a8_titanic/data/train.csv")
# use most frequent, it is general for discrete and continuos data
df = data_train.apply(lambda x:x.fillna(x.value_counts().index[0]))

print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
#print(data_train[0:10])
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#print(df[0:10])
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
selector = SelectFromModel(GradientBoostingClassifier(),threshold = "0.9 * mean").fit(X, y)
data = selector.transform(X)
print(len(X[0]))
print(len(data[0]))
print("feature importance", selector.estimator_.feature_importances_)
print("feature importance combine", np.sum(selector.estimator_.feature_importances_))

