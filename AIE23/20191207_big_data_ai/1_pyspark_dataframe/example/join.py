from pyspark import SparkContext 
sc = SparkContext()

rdd = sc.parallelize([(u'2', u'100', 2),(u'1', u'300', 1),(u'1', u'200', 1)])
rdd1 = sc.parallelize([(u'1', u'2'), (u'1', u'3')])
newRdd = rdd1.map(lambda x:(x[1],x[0])).join(rdd.map(lambda x:(x[0],(x[1],x[2]))))
result = newRdd.collect()

for i in result:
    print(i)