import csv
import random
import math
import operator


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# 扩展问题，可以采用其他的距离度量公式？
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
    loadDataset('2_knn/iris.data', split, trainingSet, testSet)
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
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()

# q1 count distance
# q2 normalize
# q3 index
# q4 sampling
