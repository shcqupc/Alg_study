import csv
import random
import math
import operator
import numpy as np


# 加载鸢尾花数据集文件并按照一定比例split拆分成训练集trainingSet和测试集testSet
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])  # 数据集前4列转换为浮点型
            if random.random() < split:  # 按传入的比例拆分数据集
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# 扩展问题，可以采用其他的距离度量公式？
# 计算标准化欧式距离
def steuDistance(instance1, instance2, length):
    tst_instance = []
    tra_instance = []
    for x in range(length):
        tst_instance.append(float(instance1[x]))
        tra_instance.append(float(instance2[x]))
    X = np.vstack([tst_instance, tra_instance])
    var = np.var(X, axis=0, ddof=1)
    try:
        dist = np.sqrt(((np.array(tst_instance) - np.array(tra_instance)) ** 2 / var).sum())
        if np.isnan(dist):
            return 0
        else:
            return dist
    except Exception as e:
        return 0


# 计算欧式距离
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        # 计算欧式距离
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    # (1) 计算测试样本和每个训练样本的欧式距离
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # dist = steuDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    # (2) 对距离进行排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # (3) 返回最近的K个邻居
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    # (1) 遍历K个最近的邻居中每个邻居
    for x in range(len(neighbors)):
        # 统计最近邻居中所有的类别标签数量
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # print('classVotes.items()',classVotes.items())
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    # 遍历每个测试集的元素，计算预测值和真实值是否相等，计算准确度
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    # 按黄金比例拆分数据集
    loadDataset('iris.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 3
    # (0) 遍历每个测试样本
    for x in range(len(testSet)):
        # (1) 对每个测试样本找到训练集中的最近的K邻居
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # (2) 统计K个邻居的类别
        result = getResponse(neighbors)
        # (3) 记录结果
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()

# q1 count distance
# q2 normalize
# q3 index
# q4 sampling
