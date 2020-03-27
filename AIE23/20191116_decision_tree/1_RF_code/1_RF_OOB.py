import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# 1）对每个样本，计算它作为oob样本的树对它的分类情况（约1/3的树）；

# 2）然后以简单多数投票作为该样本的分类结果；

# 3）最后用误分个数占样本总数的比率作为随机森林的oob误分率。

# （文献原文：Put each case left out in the construction of the kth tree down the kth tree to get a classification. In this way, a test set classification is obtained for each case in about one-third of the trees. At the end of the run, take j to be the class that got most of the votes every time case n was oob. The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. This has proven to be unbiased in many tests.）
# The out-of-bag (OOB) error is the average error for each z_i calculated using predictions from the trees that do not contain z_i in their 
# respective bootstrap sample. This allows the RandomForestClassifier to be fit and validated whilst being trained [1].
# obb error其实就是，随机森林里面某棵树，用来构建它的时候可能有n个数据没有用到，然后我们用这n个数据测一遍这棵树，然后obb error = 被分类错误数 / 总数

print(__doc__)

RANDOM_STATE = 123

# Generate a binary classification dataset.
X, y = make_classification(n_samples=500, n_features=25, random_state=RANDOM_STATE)

# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    ("RandomForestClassifier depth 10",
        RandomForestClassifier(max_depth = 10, oob_score=True)),
    ("RandomForestClassifier depth 3",
        RandomForestClassifier(max_depth = 3, oob_score=True))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 1
max_estimators = 50

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()