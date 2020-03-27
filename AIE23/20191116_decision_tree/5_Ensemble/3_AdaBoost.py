# AdaBoost Classification
# AdaBoost was perhaps the first successful boosting ensemble algorithm. 
# It generally works by weighting instances in the dataset by how easy 
# or difficult they are to classify, allowing the algorithm to pay or 
# or less attention to them in the construction of subsequent models.
# You can construct an AdaBoost model for classification using the AdaBoostClassifier class.
# The example below demonstrates the construction of 30 decision
# trees in sequence using the AdaBoost algorithm.

import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())