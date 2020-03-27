# coding=utf-8
#!/usr/bin/python
import matplotlib.pyplot as plt

#Training data set
#each element in x represents (x0,x1,x2)
x = [(0.,3) , (1.,3) ,(2.,3), (3.,2) , (4.,4), (0.,3) , (1.,3.1) ,(2.,3.5), (3.,2.1) , (4.,4.2)]
#y[i] is the output of y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
y = [95.364,97.217205,75.195834,60.105519,49.342380, 100.364,100.217205,100.195834,100.105519,12.342380]


epsilon = 0.001
#learning rate
alpha = 0.001 
diff = [0,0]
error1 = 0
error0 =0
m = len(x)

#init the parameters to zero
theta0 = 0
theta1 = 0
theta2 = 0
sum0 = 0
sum1 = 0
sum2 = 0

epoch = 0
error_array = []
epoch_array = []
while True:
    
     #calculate the parameters
    # 线性回归：hi(x) = theta0 + theta1 * x[i][1] + theta2 * x[i][2]  
    # 损失函数：(1/2) 累加 * (y - h(x)) ^ 2
    # theta = theta - 累和(  - alpha * (y - h(x))x )
    # 1. 随机梯度下降算法在迭代的时候，每迭代一个新的样本，就会更新一次所有的theta参数。
    #calculate the parameters
    # 2. 批梯度下降算法在迭代的时候，是完成所有样本的迭代后才会去更新一次theta参数
    print(m)
    for i in range(m):
        #begin batch gradient descent
        diff[0] = y[i]-( theta0 + theta1 * x[i][0] + theta2 * x[i][1] )
        sum0 = sum0 - ( -alpha * diff[0]* 1)
        sum1 = sum1 - ( -alpha * diff[0]* x[i][0])
        sum2 = sum2 - ( -alpha * diff[0]* x[i][1])
        #end  batch gradient descent
    
    theta0 = theta0 + sum0 / m;
    theta1 = theta1 + sum1 / m;
    theta2 = theta2 + sum2 / m;

    sum0 = 0
    sum1 = 0
    sum2 = 0
    #calculate the cost function
    error1 = 0
    for i in range(len(x)):
        error1 += ( y[i]-( theta0 + theta1 * x[i][0] + theta2 * x[i][1] ) )**2
        
    error1 = error1 / m
    error_array.append(error1)
    epoch_array.append(epoch)
    #if abs(error1-error0) < epsilon:
    if epoch == 2000:
        break
    else:
        error0 = error1
    epoch += 1
    print(' theta0 : %f, theta1 : %f, theta2 : %f, bgd error1 : %f, epoch: %f'%(theta0,theta1,theta2,error1,epoch))

print('Done: theta0 : %f, theta1 : %f, theta2 : %f'%(theta0,theta1,theta2))
plt.plot(epoch_array, error_array, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()