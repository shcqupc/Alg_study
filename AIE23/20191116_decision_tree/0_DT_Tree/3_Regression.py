from sklearn import tree
X = [[0, 0], [2, 2]] # 时间 估值
y = [0.5, 2.5] # 股价
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))
