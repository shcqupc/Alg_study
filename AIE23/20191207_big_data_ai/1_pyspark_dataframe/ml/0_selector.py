from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row
from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName("DataFrameExample") \
        .getOrCreate()

df = spark.createDataFrame([
    Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.2])),
    Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.0]))])

slicer = VectorSlicer(inputCol="userFeatures", outputCol="features", indices=[1, 2])

output = slicer.transform(df)

output.select("userFeatures", "features").show()