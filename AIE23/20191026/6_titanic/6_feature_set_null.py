# -*- coding: utf-8 -*-
# 加载相关模块和库
import sys
import io
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
"""
缺失值处理
观察一可得数据中存在缺失值
对于缺失值可以删除处理，但是由前面可见，这个缺失值对于结果有比较大的影响
这个可以通过赋值处理。
缺失值处理是数据处理的重要组成。
"""
print(__doc__)


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
data_train = pd.read_csv("data/train.csv")

def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

data_orig = data_train.values
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train.to_csv("data/fix_data_tai1.csv")
data = data_train.values

# 以age数据而言所有类为
data1 = data[:, 5]
attr = set(data1)
print(set(attr))

# 那么数值化方法可以给定每个类一个数值
dic = dict()
for idx, itr in enumerate(attr):
    dic[itr]=idx
n_data1 = []
for itr in data1:
    n_data1.append(dic[itr])
print("原始字符数据：", data1)
print("转换后数据：", n_data1)



