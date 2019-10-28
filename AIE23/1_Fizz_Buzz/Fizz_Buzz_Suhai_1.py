import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

'''
新问题：
面试官让写个程序来玩Prime. 这是一个游戏。玩家从1数到100，如果数字是质数，那么喊'prime'，如果不是但能被3整除就喊'fizz'，否则就直接说数字。
这个游戏玩起来就像是：1, prime, prime, 4, prime, fizz, ...
'''


# 特征工程构造特征方法：将数字1,2,3 ... 等构造特征（重要影响因素，对预测是number, 'fizz'还是"prime"有帮助的因素）
# 将每个输入的数，表示为一个特征数组（向量），这个特征数组有两个维度。

# 判断是否质数
def is_prime(num):
    if num <= 2:
        return 1
    else:
        for i in range(2, num - 1):
            if num % i == 0:
                return 99
    return 1


def feature_engineer(i):
    return np.array([is_prime(i), i % 3])


# 将需要预测的指标转换为数字方法：将数据的真实值（预测结果）number, "prime", "fizz"
# 分别对应转换为数字 0, 1, 2，这样后续能被计算机处理
def construct_sample_label(i):
    if is_prime(i) == 1:
        return np.array([1])
    elif i % 3 == 0:
        return np.array([2])
    else:
        return np.array([0])


# 生成训练集和测试集数据：我们的面试题目标是预测 1 到 100的情况
# 更加公平的预测，不让分类预测器较早的知道要预测的数据的情况，
# 我们选取101到300这个范围的数作为我们的训练集和测试集。
# Note: 语法说明。 for i in range(101, 300)代表Python中从for循环中遍历取值为i，并
# 赋值将i值输入到feature_engineer函数

# 训练集真题
x_train = np.array([feature_engineer(i) for i in range(101, 300)])
# print(x_train)
y_train = np.array([construct_sample_label(i) for i in range(101, 300)])
# print(y_train)

# 测试集期末考试试卷
x_test = np.array([feature_engineer(i) for i in range(1, 100)])
y_test = np.array([construct_sample_label(i) for i in range(1, 100)])

logistic = linear_model.LogisticRegression(C=0.1)  # 内存创建了 y = f(AX)
logistic.fit(x_train, y_train)  # 训练f，找A

# Returns the mean accuracy on the given test data and labels
# 代表模型精准程度
print('LogisticRegression train score: %f'
      % logistic.score(x_train, y_train))
print('LogisticRegression test score: %f'
      % logistic.score(x_test, y_test))

predict = logistic.predict([feature_engineer(231), feature_engineer(232), feature_engineer(101)])
print(predict)

#将预测映射到对应的结果
def out_transfer(num):
    p = logistic.predict([feature_engineer(num)])
    if p == 1:
        return "prime"
    elif p == 2:
        return "fizz"
    else:
        return str(num)

for i in range(1,100):
    print(out_transfer(i), end=' ')