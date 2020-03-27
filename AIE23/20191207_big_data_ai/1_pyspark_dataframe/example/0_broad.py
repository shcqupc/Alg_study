from pyspark import SparkContext 

sc = SparkContext()
someValue = 1
V = sc.broadcast(someValue)

def worker(element):
    element *= V.value
    print(element)
A = sc.parallelize([1,2,3]).map(worker)