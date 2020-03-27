# -*- coding:utf-8 -*- 
import os
import sys
from pyspark import SparkContext, SparkConf
from operator import add
# import os
# os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk1.8.0_231"
# os.environ["SPARK_HOME"] = "D:\spark-2.1.0-bin-hadoop2.7"
# os.environ["HADOOP_HOME"] = "D:\spark-2.1.0-bin-hadoop2.7"

# 在单机 driver节点 上执行的

sc = SparkContext("local[*]")

# RDD 
lines = sc.textFile("test.txt") # hdfs://ip:9000/test.txt

def reduce_func(a, b):
    return a + b 
    
def flat_func(x):
    return x.split(' ')
    
# 分布式的RDD上执行的
tmp = lines.flatMap(lambda x:flat_func(x))

counts = tmp.reduce(lambda a, b: reduce_func(a, b))

print("counts",counts)

sc.stop()
