# -*- coding: utf-8 -*-  
from __future__ import print_function  
from pyspark.sql import SparkSession  
from pyspark.sql import Row  
  
if __name__ == "__main__":  
    # 初始化SparkSession  
    spark = SparkSession \
        .builder \
        .appName("RDD_and_DataFrame") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
  
    sc = spark.sparkContext  
  
    lines = sc.textFile("employee.txt")  # RDD
    parts = lines.map(lambda l: l.split(","))  # ["s", "100"]
    # rdd.map(lambda r: r.features).map(lambda p: Row(name=p[0], salary=int(p[1])))
    employee = parts.map(lambda p: Row(name=p[0], salary=int(p[1])))  
  
    #RDD转换成DataFrame  
    employee_temp = spark.createDataFrame(employee)  
  
    #显示DataFrame数据 
    # (name, salary)
    # ("a", 1222) 
    employee_temp.show()  
  
    #rdd行级操作和列级操作，测试map操作符 (pyspark 2.0后取消dataframe map operator)
    employee_map_test1 = employee_temp.rdd.map(lambda x:(x,1)).take(2)  
    employee_map_test2 = employee_temp.rdd.map(lambda x: x.name).take(2)  

    #显示DataFrame数据  
    print(employee_map_test1)  
    print(employee_map_test2)  

    # map操作完成后转换为 dataframe
    temp = employee_temp.rdd.map(lambda x:(x.name,1)).map(lambda p: Row(name=p[0], tag=int(p[1])))  
    employee_df_processed= spark.createDataFrame(temp)
    employee_df_processed.show()

    # dataframe进行列级操作
    employee_df_processed.withColumn('tag_2',  employee_df_processed.tag/2.0).select('tag','tag_2').show(5)

    #创建视图 可以采用spark sql进行操作
    employee_temp.createOrReplaceTempView("employee")  
    #过滤数据  
    employee_result = spark.sql("SELECT name,salary FROM employee WHERE salary >= 14000 AND salary <= 20000")  
  
    # DataFrame转换成RDD  
    result = employee_result.rdd.map(lambda p: "name: " + p.name + "  salary: " + str(p.salary)).collect()  
  
    #打印RDD数据  
    for n in result:  
        print(n)  

# (1) 由rdd创建dataframe?
# (2) dataframe如何转化为rdd?
# (3) 如何由tag列创建新列tag * tag

