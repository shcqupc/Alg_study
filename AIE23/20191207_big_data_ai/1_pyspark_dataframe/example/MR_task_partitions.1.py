from pyspark import SparkContext 
import numpy as np

sc = SparkContext()

text_file = sc.textFile("1_pyspark_dataframe/example/employee.txt")

def map_local_func(item):
    return np.array([1, 2, 3])

def reduce_local_func(a, b):
    return a + b

counts1 = text_file.map(map_local_func) \
             .reduce(lambda a, b: reduce_local_func(a, b))

counts2 = text_file.mapPartitions(map_local_func) \
             .reduce(lambda a, b: reduce_local_func(a, b))
             
print(counts1)

print(counts2)

