from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris

iris = load_iris()
y = iris['target']
X = iris['data']
clf = RandomForestClassifier(criterion='gini', max_depth=2, n_estimators=5, oob_score=True)
clf.fit(X, y)
#print(clf.feature_importances_)
print(clf.oob_score_)
print(clf.predict([[0, 0, 0, 0]]))
