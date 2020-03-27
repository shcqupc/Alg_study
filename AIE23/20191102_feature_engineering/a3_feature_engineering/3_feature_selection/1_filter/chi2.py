
# 卡方检验
#经典的卡方检验是检验定性自变量对定性因变量的相关性。
# 卡方检验计算：
# 　　假设有两个分类变量X和Y，它们的值域分别为{x1, x2}和{y1, y2}，其样本频数列联表为：
# 参考图示(独立样本四格表)： 
# 	<x1, y1> <x2, y2> <x1, y2> <x2, y1>
# (1) a,d != 0 c,b =0
# (2) a = b =c =d 
#    y1    y2    总计
# x1   a    b  a+b
# x2   c    d  c+d
# 总计 a+c b+d a+b+c+d
# 　　若要推断的论述为H1：“X与Y有关系”，可以利用独立性检验来考察两个变量是否有关系，并且能较精确地给出这种判断的可靠程度。
# 具体的做法是，由表中的数据算出随机变量K^2的值（即K的平方）
# 　　K^2 = n (ad - bc) ^ 2 / [(a+b)(c+d)(a+c)(b+d)]　其中n=a+b+c+d为样本容量
# 　　K^2的值越大，说明“X与Y有关系”成立的可能性越大。
#     选择K个最好的特征，返回选择特征后的数据
# k Number of top features to select. The “all” option bypasses selection, for use in a parameter search.
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
iris = load_iris()
selector = SelectKBest(chi2, k=2).fit(iris.data, iris.target)
data = selector.transform(iris.data)
print(data[0:5])
print(selector.scores_)