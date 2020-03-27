from pyspark import SparkContext 

sc = SparkContext()

text_file = sc.textFile("1_pyspark_dataframe/example/test.txt")

def map_local_func(item):
    return (item, 1)

def reduce_local_func(a, b):
    return a + b

counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: map_local_func(word)) \
             .reduceByKey(lambda a, b: reduce_local_func(a, b)).collect()

for i in counts:
    print(i)

