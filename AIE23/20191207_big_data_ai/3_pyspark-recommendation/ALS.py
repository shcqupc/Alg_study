from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import os
import sys
# Path for spark source folder
# os.environ['SPARK_HOME']="D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7"
# os.environ['HADOOP_HOME']="D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7"

# # # Append pyspark  to Python Path
# sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python")
# sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python\lib/py4j-0.9-src.zip")


# Load and parse the data
conf = SparkConf() \
    .setAppName("MovieLensALS") \
    .set("spark.executor.memory", "2g") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)
data = sc.textFile("3_pyspark-recommendation/ml-1m/ratings.dat")
ratings = data.map(lambda l: l.split('::'))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
#model.save(sc, "target/tmp/myCollaborativeFilter")
#sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")