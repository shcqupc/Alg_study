import sys
from pyspark.context import SparkContext
from numpy import array, random as np_random
import numpy as np
from sklearn import linear_model as lm
from sklearn.base import copy
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.partial_fit
N = 10000   # Number of data points
D = 2      # Numer of dimensions
ITERATIONS = 5
np_random.seed(seed=42)
alpha = 0.001

def generate_data(N):
    return [[np.array([1]) if np_random.rand() < 0.5 else np.array([0]), np_random.randn(1, D)] for ii in range(N)]

def train(iterator, sgd):
    theta0 = sgd[0]
    theta1 = sgd[1]
    theta2 = sgd[2]
    sgd = [0, 0, 0]
    for x in iterator:
        print(x)
        i = 1
        y = x[0]
        x = x[1][0]
        diff = y-( theta0 * 1 + theta1 * x[0] + theta2 * x[1] )
        # - (y - h(x))x
        gradient0 = - diff* 1
        gradient1 = - diff* x[0]
        gradient2 = - diff* x[1]
        # theta = theta - (  - alpha * (y - h(x))x )
        sgd[0] +=  gradient0
        sgd[1] +=  gradient1
        sgd[2] +=  gradient2
    yield sgd

def merge(left, right):
    gradient = []
    for i in range(0,3):
        gradient.append(left[i] + right[i])
    return gradient

# def avg_model(sgd, slices):
#     sgd = np.array(sgd)
#     sgd /= slices 
#     return sgd

if __name__ == "__main__":

    sc = SparkContext()
    ITERATIONS = 3
    slices = 3
    data = generate_data(N)
    print(len(data))

    # init stochastic gradient descent
    print(data[0])
    bgd = [0,0,0]
    alpha = 0.001
    # training
    data = sc.parallelize(data, numSlices=slices).cache()
    count = data.count()
    for ii in range(ITERATIONS):
        gradient = data \
                .mapPartitions(lambda x: train(x, bgd)) \ # repartition(12)
                .reduce(lambda x, y: merge(x, y))
        #gradient = avg_model(gradient, count) # averaging weight vector => iterative parameter mixtures
        for i in range(3):
            bgd[i] = bgd[i] - alpha * gradient[i]
        print("Iteration %d:" % (ii + 1))
        print("Model: ")
        print(sgd)
        print("")