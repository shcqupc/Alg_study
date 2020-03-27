# from dataframe to dataframe
from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# from file to dataframe. NOTE: the dataframe format is a little diff with 
# pandas dataframe
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("kdd99") \
        .getOrCreate()

    # Load training data
    df = spark.read.option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").format("com.databricks.spark.csv").load("G:\\dl_data\\ctr\\train_sample.csv").cache()
    df.printSchema()
    # encoding
    indexer = StringIndexer(inputCol="site_domain", outputCol="site_domainIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.describe("site_domainIndex").show()

    # select related feature
    filtered = indexed.select("click", "site_domainIndex", "hour")
    filtered.describe().show()
    
    assembler = VectorAssembler(inputCols=["site_domainIndex", "hour"], outputCol="features")
    output = assembler.transform(filtered)
    output = output.select("features", "click")
    output.show(n = 3)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = output.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    lr = LogisticRegression(labelCol="click", featuresCol="features",maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    model = lr.fit(trainingData)    

    # Make predictions.
    predictions = model.transform(testData)
    predictions = predictions.withColumnRenamed("click", "label")
    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)
    predictionAndLabels = predictions.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    spark.stop()