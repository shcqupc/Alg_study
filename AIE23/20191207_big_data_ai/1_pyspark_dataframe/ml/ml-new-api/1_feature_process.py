# from dataframe to dataframe
from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

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
    # encoding stringindexer ~ pandas labelencoder
    indexer = StringIndexer(inputCol="site_domain", outputCol="site_domainIndex")
    indexed = indexer.fit(df).transform(df)
    indexed.describe("site_domainIndex").show()

    # select related feature
    filtered = indexed.select("click", "site_domainIndex").describe().show()
    spark.stop()