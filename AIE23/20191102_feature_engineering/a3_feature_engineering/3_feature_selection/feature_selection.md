特征选择主要有两个功能：

- 减少特征数量、降维，使模型泛化能力更强，减少过拟合

- 增强对特征和特征值之间的理解
拿到数据集，一个特征选择方法，往往很难同时完成这两个目的。通常情况下，我们经常不管三七二十一，选择一种自己最熟悉或者最方便的特征选择方法（往往目的是降维，而忽略了对特征和数据理解的目的）。
- 减少计算量


- simplification of models to make them easier to 
- interpret by researchers/users,[1]
- shorter training times,
- to avoid the curse of dimensionality,
enhanced generalization by reducing overfitting[2] (formally, reduction of variance[1])

Ref: https://en.wikipedia.org/wiki/Feature_selection

在许多机器学习相关的书里，很难找到关于特征选择的内容，因为特征选择要解决的问题往往被视为机器学习的一种副作用，一般不会单独拿出来讨论。


- Wrapper methods use a predictive model to score feature subsets. Each new subset is used to train a model, which is tested on a hold-out set. Counting the number of mistakes made on that hold-out set (the error rate of the model) gives the score for that subset. As wrapper methods train a new model for each subset, they are very computationally intensive, but usually provide the best performing feature set for that particular type of model.
Recursive Feature Elimination algorithm, commonly used with Support Vector Machines to repeatedly construct a model and remove features with low weights.

- Filter Filters are usually less computationally intensive than wrappers, but they produce a feature set which is not tuned to a specific type of predictive model.[7] This lack of tuning means a feature set from a filter is more general than the set from a wrapper, usually giving lower prediction performance than a wrapper. However the feature set doesn't contain the assumptions of a prediction model, and so is more useful for exposing the relationships between the features. Many filters provide a feature ranking rather than an explicit best feature subset, and the cut off point in the ranking is chosen via cross-validation. Filter methods have also been used as a preprocessing step for wrapper methods, allowing a wrapper to be used on larger problems.
  
- Embedded methods are a catch-all group of techniques which perform feature selection as part of the model construction process. The exemplar of this approach is the LASSO method for constructing a linear model, which penalizes the regression coefficients with an L1 penalty, shrinking many of them to zero. Any features which have non-zero regression coefficients are 'selected' by the LASSO algorithm. Improvements to the LASSO include Bolasso which bootstraps samples,;[8] Elastic net regularization, which combines the L1 penalty of LASSO with the L2 penalty of Ridge Regression; and FeaLect which scores all the features based on combinatorial analysis of regression coefficients.[9] These approaches tend to be between filters and wrappers in terms of computational complexity.


In traditional statistics, the most popular form of feature selection is stepwise regression, which is a wrapper technique. It is a greedy algorithm that adds the best feature (or deletes the worst feature) at each round. 