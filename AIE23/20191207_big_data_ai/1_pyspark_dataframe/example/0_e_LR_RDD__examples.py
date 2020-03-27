from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SparkSession  
from pyspark.sql import Row  
  
# 初始化SparkSession  
spark = SparkSession \
    .builder \
    .appName("RDD_and_DataFrame") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sc = spark.sparkContext  

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("1_pyspark_dataframe/ml/data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))

trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))
