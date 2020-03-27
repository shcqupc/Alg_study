# -*- coding: utf-8 -*-
"""
数据预处理
如前面所说，我们的数据预处理工作占用了我们的70%时间
其完成质量直接影响最终结果
首先需要对数据有个整体的认识
"""
# 加载相关模块和库
import sys
import io
from sklearn.preprocessing import Imputer
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
print(__doc__)
import pandas as pd
data_train = pd.read_csv("a8_titanic/data/train.csv")
print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
print(data_train[0:1])
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
print(df[0:10])
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df[0:1])
print("看列名", train_df.columns)

