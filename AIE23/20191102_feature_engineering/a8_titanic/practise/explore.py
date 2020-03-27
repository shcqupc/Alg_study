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
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data_train = pd.read_csv("a8_titanic/data/train.csv")
print("看列名", data_train.columns)
# 数据摸底
print("看每列性质，空值和类型", data_train.info())
# 问题1 空值填充
# 问题2 类别型数据处理
print("看每列统计信息", data_train.describe())
# 3 看类别不平衡的处理
print("看类别平衡的状况\n", data_train['Survived'].value_counts())

#
# PassengerId => 乘客ID    
# Survived => 获救情况（1为获救，0为未获救）  
# Pclass => 乘客等级(1/2/3等舱位) 
# Name => 乘客姓名     
# Sex => 性别     
# Age => 年龄     
# SibSp => 堂兄弟/妹个数     
# Parch => 父母与小孩个数 
# Ticket => 船票信息     
# Fare => 票价     
# Cabin => 客舱     
# Embarked => 登船港口 
df = data_train
train_df = df.filter(regex='Survived|SibSp|Parch|Fare|Cabin|Name')
#train_df.to_csv("processed_titanic.csv" , encoding = "utf-8")
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)