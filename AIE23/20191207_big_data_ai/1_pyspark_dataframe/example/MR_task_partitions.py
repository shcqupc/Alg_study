from pyspark import SparkContext 

sc = SparkContext()

text_file = sc.textFile("1_pyspark_dataframe/example/test.txt")
num = 1
def map_local_func(item):
    c = num + 1
    result = []
    for i in item:
        result.append((i, 1))
    return result

def reduce_local_func(a, b):
    return a + b

counts = text_file.mapPartitions(map_local_func) \
             .reduceByKey(lambda a, b: reduce_local_func(a, b)).collect()

for i in counts:
    print(i)

