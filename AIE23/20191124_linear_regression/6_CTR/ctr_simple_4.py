import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split

data_train = pd.read_csv('D:\\dl_data\\ctr\\train_sample.csv')
    
train_df = data_train.filter(regex='click|hour|C1|banner_pos')
print(train_df.describe())
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
print(y)
# X即特征属性值
X = train_np[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg = linear_model.LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

clf = linear_model.LogisticRegression(C=0.001,penalty="l2")
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))