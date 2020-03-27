# -*- coding: utf-8 -*-
import pandas as pd #数据分析
# data_train = pd.read_csv("D:\\project\\peixun\\ai_course_project_px\\1_intro\\4_anli_project_titanic\\Kaggle_Titanic_Chinese\\Kaggle_Titanic-master\\train.csv")
# # (1) 看列名
# print(data_train.columns)

# # (2) 看每列性质，空值和类型
# print(data_train.info())

# # (3) 看每列统计信息
# print(data_train.describe())

data_train = pd.read_csv("D:\\dataset\\ai_course_data\\ctr\\all\\train\\train.csv")
print(data_train.columns)
sample_train = data_train.sample(10000)
sample_train.to_csv("D:\\dataset\\ai_course_data\\ctr\\sample\\ctr_train_sample.csv", index = False)

data_test = pd.read_csv("D:\\dataset\\ai_course_data\\ctr\\all\\test\\test.csv")
print(data_test.columns)
sample_test = data_test.sample(10000)
sample_test.to_csv("D:\\dataset\\ai_course_data\\ctr\\sample\\ctr_test_sample.csv", index = False)

