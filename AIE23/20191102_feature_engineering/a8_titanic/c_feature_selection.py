# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.learning_curve import learning_curve
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd #数据分析
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from util import set_Cabin_type, set_missing_ages, plot_learning_curve, one_hot_encoding
# (1) 读取数据集
data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")

# (2) 特征工程 - 处理缺失值
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

# (3) 特特工程 - 类目型的特征离散/因子化
df = one_hot_encoding(data_train)
# select specific coloumn
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#print(train_df.describe())
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]


# (5) 模型构建与训练
clf = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=5, verbose=2)
#clf.fit(X, y)
#print(clf.predict(y))
# (6) 绘制learning curve
plot_learning_curve(clf, u"学习曲线", X, y)


# (5) 模型构建与训练 - 特征消除
# 两者对比，对提升测试集精度有效果，适当减轻了过拟合。测试保留6个
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
selector = RFE(estimator=LogisticRegression(), n_features_to_select=6).fit(X, y)
X = selector.transform(X)
print(X)
print(selector.ranking_)
clf = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=5, verbose=0)
# (6) 绘制learning curve
plot_learning_curve(clf, u"学习曲线-特征消除", X, y)