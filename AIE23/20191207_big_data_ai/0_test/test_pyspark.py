# -*- coding:utf-8 -*- 
import os
import sys
# Path for spark source folder
#os.environ['SPARK_HOME']="D:\\data_and_dep\\sparkml\\software\\spark-2.2.0-bin-hadoop2.7\\spark-2.2.0-bin-hadoop2.7"
#os.environ['JAVA_HOME']="C:\\Program Files\\Java\\jre1.8.0_151"
#os.environ['HADOOP_HOME']="D:\\data_and_dep\\sparkml\\software\\spark-2.2.0-bin-hadoop2.7\\spark-2.2.0-bin-hadoop2.7"
# # # Append pyspark  to Python Path
#sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python")
#sys.path.append("D:\data_and_dep\sparkml\software\spark-2.1.0-bin-hadoop2.7\spark-2.1.0-bin-hadoop2.7\python\lib/py4j-0.9-src.zip")
from pyspark import SparkContext, SparkConf
from operator import add


# java gateway error: https://blog.csdn.net/a2099948768/article/details/79580634
# https://blog.csdn.net/xiaoshunzi111/article/details/78834615
# make java path no blank.

sc = SparkContext("local[*]") # SparkSession

def add_2(a, b):
    "Same as a + b."
    return a + b 

lines = sc.textFile("test.txt") # hdfs://
tmp = lines.flatMap(lambda x:x.split(' ')).map(lambda x:(x,1))
counts = tmp.reduceByKey(add_2) # group by key
output = counts.collect() # count / take  === action 
#counts.count()
for (word,count) in output:
    print("xxx: %s %i" % (word,count))

sc.stop()
