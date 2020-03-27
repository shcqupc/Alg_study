# 因为LR本身的模型复杂度不够，很多问题不仅仅是线性问题，所以通过特征组合，希望线性模型拟合更复杂的的问题
# 基于tf https://segmentfault.com/a/1190000014799038?utm_source=tag-newest
# UDF apply可以https://blog.csdn.net/kwame211/article/details/78109254
# for feature in features:
#     file = filePath + feature + ".csv"
#     if (os.path.isfile(file)):
#         lines = open(file).readlines()
#         for line in lines:
#             key = line.split(',')[0]
#             df[feature+'_'+key] = df[feature].apply(lambda x: 1 if x == key else 0)

# apply is flexible, could combine with headers or column names
# need test to check such as iris.
#  def feature_combine2(feature1, feature2, df)
#  for f1 in  feature1 values
#    for f2 in feature2 values
#        f1index
#        f2index
#      df[feature+'_'+key] = df[feature].apply(lambda x: 1 if x[f1index] == f1 and x[f2index] == f2 else 0)