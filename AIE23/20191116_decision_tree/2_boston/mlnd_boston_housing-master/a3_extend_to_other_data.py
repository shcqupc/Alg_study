# 载入此项目所需要的库
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 载入波士顿房屋的数据集
data = pd.read_csv('2_boston/mlnd_boston_housing-master/bj_housing.csv')
prices = data['Value']
features = data.drop('Value', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=0)

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

predicted_price = regressor.predict(X_test)

# 提示：你可能需要参考问题3的代码来计算R^2的值
def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
    return score

r2 = performance_metric(y_test, predicted_price)

print("Optimal model has R^2 score {:,.2f} on test data".format(r2))

