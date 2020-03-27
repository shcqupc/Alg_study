import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

x_train = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv")
print(x_train.info())
"""
Age            714 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
"""


def set_missing_ages(df):
    # put numerical feature into Random Forest Regressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    print("df.Age.isnull", df.PassengerId[df.Age.isnull()].count())
    # print(age_df[age_df.Age.isnull()].count())

    # split age_df into known age and unknown age
    known_age = age_df[age_df.Age.notnull()].values
    unknow_age = age_df[age_df.Age.isnull()].values
    print("type(known_age)", type(known_age))
    # print(known_age.shape, unknow_age.shape)
    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(x, y)
    print(rfr.score(x, y))
    predictage = rfr.predict(unknow_age[:, 1:])
    # assign predicted age to df
    df.loc[df.Age.isnull(), "Age"] = predictage
    print("df.Age.isnull", df.PassengerId[df.Age.isnull()].count())
    return df, rfr


def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(), "Cabin"] = 'Y'
    df.loc[df.Cabin.isnull(), "Cabin"] = 'N'
    # print(df.groupby(["Cabin"]).count().PassengerId)
    return df

def dummies(df):
    # 'Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'
    dummies_Cabin = pd.get_dummies(x_train.Cabin, prefix="Cabin")
    # print(dummies_Cabin)
    dummies_Pclass = pd.get_dummies(x_train.Pclass, prefix="Pclass")
    dummies_Sex = pd.get_dummies(x_train.Sex, prefix="Sex")
    dummies_Embarked = pd.get_dummies(x_train.Embarked, prefix="Embarked")
    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df

print('\n---------------set_missing_ages---------------------')
x_train, rfr = set_missing_ages(x_train)
print('\n---------------set_Cabin_type---------------------')
x_train = set_Cabin_type(x_train)
# print(x_train.loc[x_train.Embarked.isnull(),"Embarked"])
# x_train.loc[x_train.Embarked.isnull(), "Embarked"] = "Q"

print('\n---------------OneHotEncoder---------------------')
# from sklearn.preprocessing import OneHotEncoder
# print(x_train[["Sex", "Embarked", "Cabin"]][0:5])
# enc = OneHotEncoder()
# x_train_age = enc.fit_transform(x_train[["Sex", "Embarked", "Cabin"]]).toarray()
# print(x_train_age)

print('\n---------------get_dummies---------------------')
x_train = dummies(x_train)
print(x_train.columns)

print('\n---------------filter---------------------')
train_df = x_train.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df.columns)
train_np = train_df.values
print(type(train_np))

print('\n--------------training----------------------')
import xgboost as xgb
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"测试集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

x = train_np[:,1:]
y= train_np[:,0]

clf = xgb.XGBClassifier(max_depth = 3, n_estimators = 5)
# (6) 绘制learning curve
plot_learning_curve(clf, u"学习曲线", x, y)


