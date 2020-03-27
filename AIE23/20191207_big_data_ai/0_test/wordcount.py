
#!/usr/bin/env python
#-*-conding:utf-8-*-
 
import logging
from operator import add
from pyspark import SparkContext
 
logging.basicConfig(format='%(message)s', level=logging.INFO)  
 
#import local file
test_file_name = "README.md"  # hdfs://
out_file_name = "result.txt"
 
sc = SparkContext("local", "wordcount app")
 
# text_file rdd object
text_file = sc.textFile(test_file_name)
#print(test_file.take(1))
# counts
counts = text_file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile(out_file_name)
test = counts.count()