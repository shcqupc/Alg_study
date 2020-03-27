from pyspark import SparkContext 

sc = SparkContext()

text_file = sc.textFile("1_pyspark_dataframe/example/employee.txt")

rdd = text_file.map(lambda line: line.split(",")) \
             .map(lambda item: (item[0], item[1])) \
             .reduceByKey(lambda a, b: a + b)

counts = rdd.collect()


f = open("result.txt", "w")

for i in counts:
    print(i)
    f.writelines(str(i) + "\n")

f.close()

# for cluster: rdd.saveAsTextFile()