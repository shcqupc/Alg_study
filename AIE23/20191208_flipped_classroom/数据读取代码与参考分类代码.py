import os 
import numpy as np 
import jieba  

class Data():
    def __init__(self, train_dir="Train", test_dir="Test"): 
        """
        初始化
        输入参数：训练、测试数据文件
        """
        self.train_data, self.train_label = self.process_dirs(train_dir)
        self.test_data, self.test_label = self.process_dirs(test_dir)
        print(len(self.train_data), len(self.train_label))
    def process_dirs(self, base): 
        """
        读取数据
        输入参数：数据文件夹
        """
        if os.path.exists(base+".npz")==True: 
            files = np.load(base+".npz")
            inputs = files["inputs"] 
            labels = files["labels"] 
            return inputs, labels 
        pos_dir = os.path.join(base, "pos")
        pos_files = os.listdir(pos_dir) 
        neg_dir = os.path.join(base, "neg")
        neg_files = os.listdir(neg_dir) 
        labels = [] 
        inputs = []
        for itr_name in pos_files:
            files = open(os.path.join(pos_dir, itr_name), "r", encoding="gbk", errors="ignore")
            file_data = files.read() 
            data_seg = " ".join(jieba.lcut(file_data))
            inputs.append(data_seg)
            labels.append(0) 
        for itr_name in neg_files:
            files = open(os.path.join(neg_dir, itr_name), "r", encoding="gbk", errors="ignore")
            file_data = files.read() 
            data_seg = " ".join(jieba.lcut(file_data))
            inputs.append(data_seg)
            labels.append(1)  
        inputs = np.array(inputs) 
        labels = np.array(labels)  
        np.savez(base+".npz", inputs=inputs, labels=labels)
        return inputs, labels 
import time 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.svm import SVC  
from sklearn.naive_bayes import MultinomialNB 
# 数据读取类
data_tool = Data()
# 向量化类 
vect_tool = CountVectorizer()  
# 文本降维类
lda_tool = LatentDirichletAllocation(n_components=64) 
# 文本分类
svm_tool = SVC() 
mnb_tool = MultinomialNB() 


# 贝叶斯处理稀疏数据分类过程
## 向量化
vect_sparse = vect_tool.fit_transform(data_tool.train_data) 
## 分类
vect_dense = mnb_tool.fit(vect_sparse, data_tool.train_label) 
# 预测过程 
timestart = time.perf_counter() 
test_sparse = vect_tool.transform(data_tool.test_data) 
pred = mnb_tool.predict(test_sparse) 
timeend = time.perf_counter() 
acc = np.sum(pred==data_tool.test_label)/len(pred)
print("MNB Time:{:.2f}, Acc:{:.2f}".format(timeend-timestart, acc))


# SVM处理降维后数据分类过程
## 向量化
vect_sparse = vect_tool.fit_transform(data_tool.train_data) 
## 文本降维
vect_dense = lda_tool.fit_transform(vect_sparse) 
## 分类 
svm_tool.fit(vect_dense, data_tool.train_label) 
# 预测过程 
timestart = time.perf_counter() 
test_sparse = vect_tool.transform(data_tool.test_data) 
test_dense = lda_tool.transform(test_sparse) 
pred = svm_tool.predict(test_dense) 
timeend = time.perf_counter() 
acc = np.sum(pred==data_tool.test_label)/len(pred)
print("LDA+SVM Time:{:.2f}, Acc:{:.2f}".format(timeend-timestart, acc))






            