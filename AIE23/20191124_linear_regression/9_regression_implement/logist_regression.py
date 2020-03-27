# -*- coding:utf-8 -*-  
#功能： 使用tensorflow实现一个简单的逻辑回归  
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
  
#创建占位符  
X=tf.placeholder(tf.float32)  
Y=tf.placeholder(tf.float32)  
  
#创建变量  
#tf.random_normal([1])返回一个符合正态分布的随机数  
w=tf.Variable(tf.random_normal([1],name='weight'))  
b=tf.Variable(tf.random_normal([1],name='bias'))  
  
y_predict=tf.sigmoid(tf.add(tf.multiply(X,w),b))  
num_samples=400  
cost=tf.reduce_sum(tf.pow(y_predict-Y,2.0))/num_samples  
  
#学习率  
lr=0.01  
optimizer=tf.train.AdamOptimizer().minimize(cost)  
  
#创建session 并初始化所有变量  
num_epoch=500  
cost_accum=[]  
cost_prev=0  
#np.linspace（）创建agiel等差数组，元素个素为num_samples  
xs=np.linspace(-5,5,num_samples)  
ys=np.sin(xs)+np.random.normal(0,0.01,num_samples)  
  
with tf.Session() as sess:  
    #初始化所有变量  
    sess.run(tf.initialize_all_variables())  
    #开始训练  
    for epoch in range(num_epoch):  
        for x,y in zip(xs,ys):  
            sess.run(optimizer,feed_dict={X:x,Y:y})  
        train_cost=sess.run(cost,feed_dict={X:x,Y:y})  
        cost_accum.append(train_cost)  
        print("train_cost is:",str(train_cost))  
  
        #当误差小于10-6时 终止训练  
        if np.abs(cost_prev-train_cost)<1e-6:  
            break  
        #保存最终的误差  
        cost_prev=train_cost  
#画图  画出每一轮训练所有样本之后的误差  
plt.plot(range(len(cost_accum)),cost_accum,'r')  
plt.title('Logic Regression Cost Curve')  
plt.xlabel('epoch')  
plt.ylabel('cost')  
plt.show()  