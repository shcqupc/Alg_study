from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
# 递归特征消除法，返回特征选择后的数据
# 递归特征消除法
# 递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用feature_selection库的RFE类来选择特征的代码如下：
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数7
print(iris.data[0:5])
selector = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data[0:5])
print(selector.ranking_)