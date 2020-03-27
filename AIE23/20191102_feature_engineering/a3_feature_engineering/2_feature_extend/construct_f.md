1 两个会出现多次的列进行组合
相乘，group by count其中一列
由于儿童更能得到照顾，所以构造特征：

data_train['isChild']=(data_train['Age']<=10).astype(int)

年龄越大越不易生存，所以放大年龄：

data_train['Age']=data_train['Age']*data_train['Age']

考虑几等舱（比如一等）和年轻的更容易生存，故构造特征Age∗Class：

data_train['Age_Pclass']=data_train['Age']*data_train['Pclass']

2 单属性可以扩展
排序（在这列中的排名），计数（两列group by一列），比率（），单独二值或者单值化，两列乘组合新特征


1 用户在考察日前n天的行为总数计数	反映了user_id的活跃度（不同时间粒度：最近1天/3天/6天）
2 用户的点击购买转化率	反映了用户的购买决策操作习惯
3 用户的点击购买平均时差	反映了用户的购买决策时间习惯
4 商品在所属类别中的行为总数排序	反映了item_id在item_category中的热度排名（用户停留性）