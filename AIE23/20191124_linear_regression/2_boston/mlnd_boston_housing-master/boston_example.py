# 载入此项目所需要的库
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code

# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#TODO 1

#目标：计算价值的最小值
minimum_price = np.min(prices)

#目标：计算价值的最大值
maximum_price = np.max(prices)

#目标：计算价值的平均值
mean_price = np.mean(prices)

#目标：计算价值的中值
median_price = np.median(prices)

#目标：计算价值的标准差
std_price = np.std(prices)

#目标：输出计算的结果
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

import matplotlib
rm = data['RM']
medv = data['MEDV']
matplotlib.pyplot.scatter(rm, medv, c='b')
matplotlib.pyplot.show()
lstat = data['LSTAT']
matplotlib.pyplot.scatter(lstat, medv, c='c')
matplotlib.pyplot.show()
ptratio = data['PTRATIO']
matplotlib.pyplot.scatter(ptratio, medv, c='g')
matplotlib.pyplot.show()

# TODO 2
# 提示： 导入train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=0)

print("Train test split success!")

# TODO 3

# 提示： 导入r2_score
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
    return score

#提示: 导入 'KFold' 'DecisionTreeRegressor' 'make_scorer' 'GridSearchCV' 
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# TODO 4
#提示: 导入 'KFold' 'DecisionTreeRegressor' 'make_scorer' 'GridSearchCV' 
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

""" 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""
# K折交叉验证
cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)
# 决策树回归模型
regressor = DecisionTreeRegressor()

params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

# 将R2作为评价指标
scoring_fnc = make_scorer(performance_metric)

grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator)

# 基于输入数据 [X,y]，进行网格搜索
grid = grid.fit(X_train, y_train)

print("best param" + str(grid.best_params_))
print("best score" + str(grid.best_score_))

# 基于训练数据，获得最优模型
optimal_reg =  grid.best_estimator_ 

# 输出最优模型的 'max_depth' 参数
print("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))

# 生成三个客户的数据
client_data = [[5, 17, 15], # 客户 1
               [4, 32, 22], # 客户 2
               [8, 3, 12]]  # 客户 3

# 进行预测
predicted_price = optimal_reg.predict(client_data)
for i, price in enumerate(predicted_price):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

# 提示：你可能需要用到 X_test, y_test, optimal_reg, performance_metric
predicted_price = optimal_reg.predict(X_test)
# 提示：可能需要代码来计算R^2的值
r2 = performance_metric(y_test, predicted_price)
print("Optimal model has R^2 score {:,.2f} on test data".format(r2))



