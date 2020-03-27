import numpy as np
import pandas as pd
import visuals as vs
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""

    score = r2_score(y_true, y_predict, sample_weight=None, multioutput=None)
    return score
    
def fit_model(X, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""
    
    cross_validator = KFold(n_splits=10, shuffle=False, random_state=None)    
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cross_validator, verbose=1)
    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)
    # 返回网格搜索后的最优模型
    return grid.best_estimator_

# Load the Beijing housing dataset
# 载入北京房屋的数据集
data = pd.read_csv('bj_housing2.csv')
prices = data['Value']
features = data.drop('Value', axis=1)
print("Beijing housing dataset has {} data points with {} variables each.".format(*data.shape))

# 观察原始数据，决定是否进行预处理
value = data['Value']
for item in ['Area', 'Room', 'Living', 'Year', 'School', 'Floor']:
    idata = data[item]
    matplotlib.pyplot.scatter(idata, value)
    matplotlib.pyplot.show()

#计算价值的最小值
minimum_price = np.min(prices)
#计算价值的最大值
maximum_price = np.max(prices)
#计算价值的平均值
mean_price = np.mean(prices)
#计算价值的中值
median_price = np.median(prices)
#计算价值的标准差
std_price = np.std(prices)
#输出计算的结果
print("Statistics for Beijing housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=0)
print("Train test split success!")

# 根据不同的训练集大小，和最大深度，生成学习曲线
vs.ModelLearning(X_train, y_train)
# 根据不同的最大深度参数，生成复杂度曲线
vs.ModelComplexity(X_train, y_train)

# 基于训练数据，获得最优模型
optimal_reg = fit_model(X_train, y_train)
# 输出优模型的 'max_depth' 参数
print("Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth']))

# 预测
predicted_price = optimal_reg.predict(X_test)
# 计算R^2的值
r2 = performance_metric(predicted_price, y_test)
print("Optimal model has R^2 score {:,.2f} on test data".format(r2))