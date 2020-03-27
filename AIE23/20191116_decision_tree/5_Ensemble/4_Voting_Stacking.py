# Voting is one of the simplest ways of combining the predictions 
# from multiple machine learning algorithms.
# It works by first creating two or more standalone models from 
# your training dataset. A Voting Classifier can then be used to 
# wrap your models and average the predictions of the sub-models 
# when asked to make predictions for new data.
# The predictions of the sub-models can be weighted, but specifying
# the weights for classifiers manually or even heuristically is 
# difficult. More advanced methods can learn how to best weight the
# predictions from submodels, but this is called stacking (stacked aggregation) 
# and is currently not provided in scikit-learn.
# You can create a voting ensemble model for classification using 
# the VotingClassifier class.
# The code below provides an example of combining the predictions 
# of logistic regression, classification and regression trees and 
# support vector machines together for a classification problem.
# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())