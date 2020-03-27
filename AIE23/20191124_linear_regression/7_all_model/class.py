# -*- coding: utf-8 -*-  
  
import numpy as np    
from sklearn.cross_validation import KFold  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier   
from sklearn import svm  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  

  
COLOUR_FIGURE = False    
  
      
def accuracy(test_labels, pred_lables):    
    correct = np.sum(test_labels == pred_lables)    
    n = len(test_labels)    
    return float(correct) / n    
  
#------------------------------------------------------------------------------  
#逻辑回归  
#------------------------------------------------------------------------------  
def testLR(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = LogisticRegression()  
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
  
#------------------------------------------------------------------------------  
#朴素贝叶斯  
#------------------------------------------------------------------------------  
def testNaiveBayes(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = GaussianNB()  
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
  
  
#------------------------------------------------------------------------------  
#K最近邻  
#------------------------------------------------------------------------------  
def testKNN(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = KNeighborsClassifier(n_neighbors=5)   
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
               
#------------------------------------------------------------------------------  
#--- 支持向量机  
#------------------------------------------------------------------------------  
def testSVM(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = svm.SVC()  
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
  
#------------------------------------------------------------------------------  
#--- 决策树  
#------------------------------------------------------------------------------  
def testDecisionTree(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = DecisionTreeClassifier()  
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
      
#------------------------------------------------------------------------------  
#--- 随机森林  
#------------------------------------------------------------------------------  
def testRandomForest(features, labels):  
    kf = KFold(len(features), n_folds=3, shuffle=True)    
    clf = RandomForestClassifier()  
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]    
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]    
    print(score)  
  
      
if __name__ == '__main__': 
    from sklearn.datasets import load_iris
    data = load_iris()
    features, labels = data.data, data.target
    print(features)  
      
    print('LogisticRegression: \r')  
    testLR(features, labels)  
      
    print('GaussianNB: \r')  
    testNaiveBayes(features, labels)  
      
    print('KNN: \r')  
    testKNN(features, labels)  
      
    print('SVM: \r')  
    testSVM(features, labels)  
      
    print('Decision Tree: \r')  
    testDecisionTree(features, labels)  
      
    print('Random Forest: \r')  
    testRandomForest(features, labels)  
      