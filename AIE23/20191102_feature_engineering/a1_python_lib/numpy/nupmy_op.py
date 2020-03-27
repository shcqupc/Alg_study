# 一.一维矩阵的加，减,平方，三角函数
import  numpy as np
a=np.array([10,20,30,40])
b=np.arange(4)#0,1,2,3
c=b**2
d=np.sin(a)
e=np.cos(a)
f=np.tan(a)
print(a+b)
print(a-b)
print(c)
print(d)
print(e)
print(f)
print(b<3)#返回Ture或者False,bool类型的矩阵

# 二.多维矩阵的乘法
import  numpy as np
a=np.array([[1,1],
            [0,1]])
b=np.arange(4).reshape((2,2))
c=a*b#两个同型矩阵对应元素的乘积
c_dot=np.dot(a,b)#矩阵的乘法运算
c_dot_2=a.dot(b) #矩阵ab的乘积
print(c)
print(c_dot)
print(c_dot_2)

# 三.多维矩阵行列运算
import  numpy as np     
a=np.array([[1,2,3],[2,3,4]])#shape=2x4
print(a)                
print(np.sum(a)) #15       
print(np.max(a)) #4      
print(np.min(a)) #1       
print(np.sum(a,axis=1)) #行求和[6,9]
print(np.sum(a,axis=0)) #列求和[3,5,7]
print(np.max(a,axis=0)) #列最大[2,3,4]
print(np.min(a,axis=1)) #行最小[1,2]

# 四.矩阵的索引运算 
import  numpy as np      
A=np.arange(2,14).reshape
print(A)                 
print(np.argmin(A))  #0  
print(np.argmax(A))  #11 
print(np.mean(A))  # 均值7.5 
print(A.mean())    #均值 7.5  
print(np.average(A))# 均值7.5
print(np.median(A))  #中位数7.5    
print(np.cumsum(A)) # 累加[ 2  5  9 14 20 27 35 44 54 65 77 90]
print(np.diff(A))#下面显示
#将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵
print(np.nonzero(A))   

# 五.矩阵的运算 
import  numpy as np
A=np.arange(14,2,-1).reshape(3,4)#从14到3，步长为-1
print(A)
print(np.sort(A))  #每一行排序  

# 六.矩阵的转置有两种表示方法

import  numpy as np
A=np.arange(1,10).reshape(3,3)
print(A)
print(np.transpose(A))
print(A.T)

# 七.矩阵截取clip
import numpy as np
a=np.arange(1,13).reshape((3,4))
print(a)
print(np.clip(a,5,9))#最小5，最大9，小于5的都成了5，大于9的都成了9
