# Random Forest Classification
# Random forest is an extension of bagged decision trees.
# Samples of the training dataset are taken with replacement,
# but the trees are constructed in a way that reduces the 
# correlation between individual classifiers. 
# Specifically, rather than greedily choosing the best split 
# point in the construction of the tree, only a random subset of features are considered for each split
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())