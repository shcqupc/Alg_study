import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
# names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
# array = dataframe.values

from sklearn.datasets import load_boston
boston = load_boston()

# X = array[:,0:13]
# Y = array[:,13]

X = boston.data
Y = boston.target

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
def bulid_model(model_name):
    model = model_name()
    return model
scoring = 'mean_squared_error'

for model_name in [LinearRegression,Ridge,Lasso,ElasticNet,KNeighborsRegressor,DecisionTreeRegressor,SVR]:
    import time
    starttime = time.clock()
    model = bulid_model(model_name)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    print(model_name)
    print(results.mean())
    #long running
    endtime = time.clock()
    print('Running time: %s Seconds'%(endtime-starttime))

