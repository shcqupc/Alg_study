# 载入此项目所需要的库
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# 载入波士顿房屋的数据集
data = pd.read_csv('2_boston/mlnd_boston_housing-master/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=0)

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

predicted_price = regressor.predict(X_test)

print(predicted_price)


