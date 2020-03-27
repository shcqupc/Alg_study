from __future__ import print_function
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

# from file to dataframe. NOTE: the dataframe format is a little diff with 
# pandas dataframe
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("test") \
        .getOrCreate()

    # Load training data
    df = spark.read.option("header", "true").option("inferschema", "true").option("mode", "DROPMALFORMED").format("com.databricks.spark.csv").load("G:\\dl_data\\ctr\\train_sample.csv").cache()
    print("Schema from csv:")
    df.printSchema()
    print("Loaded training data as a DataFrame with " +
          str(df.count()) + " records.")
    # Show statistical summary of labels.
    df.show(truncate=3, n=3)   
    df.describe("click").show()
    #df.describe().show()
    spark.stop()