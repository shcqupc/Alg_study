#scikit-learn已经有了模型持久化的操作，导入joblib即可
from sklearn.externals import joblib
#模型保存
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y) 
joblib.dump(clf, "train_model.m")

# 通过joblib的dump可以将模型保存到本地，clf是训练的分类器
# 模型从本地调回
clf = joblib.load("train_model.m")
# 通过joblib的load方法，加载保存的模型。
# 然后就可以在测试集上测试了
clf.predict(X)

