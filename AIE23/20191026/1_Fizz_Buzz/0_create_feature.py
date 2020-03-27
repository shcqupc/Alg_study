import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
# 分类，回归（有监督）和聚类（无监督）
# 1 数据预处理/特征工程
# 样本 train test
# 单样本 x(特征),y(标签，回归分类才有)
# 2 模型训练
# 模型 
# 损失函数
# 评测指标 train metric
# 3 模型预测
# test metric
# one, two, three
# def input_transfor(num_str):
#     if num_str == "one": return 1;
#     # ... 
#     return 
# 特征工程构造特征方法：将数字1,2,3 ... 等构造特征（重要影响因素，对预测number, "fizz", "buzz", "fizzbuzz"有帮助的因素），构造为三个维度。
# 将每个输入的数，表示为一个特征数组（向量），这个特征数组有三个维度。
def feature_engineer(i):
    #return np.array([i])
    return np.array([i % 3, i % 5, i % 15])
    #return np.array([i % 3, i % 5])


# 将需要预测的指标转换为数字方法：将数据的真实值（预测结果）number, "fizz", "buzz", "fizzbuzz"
# 分别对应转换为数字 3, 2, 1, 0，这样后续能被计算机处理
def construct_sample_label(i):
    if   i % 15 == 0: return np.array([3])
    elif i % 5  == 0: return np.array([2])
    elif i % 3  == 0: return np.array([1])
    else:             return np.array([0])

#[1, 1] [2, 2] [3, fizz]
# 生成训练集和测试集数据：我们的面试题目标是预测 1 到 100的fizz buzz情况. 所以为了
# 更加公平的预测，不让分类预测器较早的知道要预测的数据的情况，
# 我们选取101到200这个范围的数作为我们的训练集和测试集。 
# Note: 语法说明。 for i in range(101, 200)代表Python中从for循环中遍历取值为i，并
# 赋值将i值输入到feature_engineer函数

#训练集真题
# [[0,1,2], [2,3,1],[1,2,3]]
x_train = np.array([feature_engineer(i) for i in range(101, 200)])
#print(x_train)
# [[1],[2],[0]]
y_train = np.array([construct_sample_label(i) for i in range(101, 200)])
#print(y_train)

#测试集期末考试试卷
x_test = np.array([feature_engineer(i) for i in range(1, 100)])
y_test = np.array([construct_sample_label(i) for i in range(1, 100)])