## 这里例子用于曲线拟合
# y=ax+b
# 如何通过tensorflow计算a,b
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
x = tf.placeholder(tf.float32, [1])
d = tf.placeholder(tf.float32, [1])
# a,b是需要不断改变的量
a = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
#这个就是函数y=ax+b
y = x * a + b
#定义loss函数
loss = tf.square(y-d)
#定义梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.05)
#定义迭代优化目标
train_step = optimizer.minimize(loss)
print("train_step",train_step)

#定义会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
dts = np.load("../data/basic-data.npz")
tx = dts['x']
ty = dts['y']
print(dts['true'])

#产生训练样本
tx = np.random.random([1000, 1])
ty = tx + 1
for itr, (x_in, y_in) in enumerate(zip(tx, ty)):
    #x_in = np.random.random([1])
    #y_in = 2 * x_in + 1
    sess.run(train_step, feed_dict={x: x_in, d:y_in})
    if(itr%100==0):
        #输出三个量：损失函数，a的值，b的值
        out = sess.run([loss, a.value(), b.value()], feed_dict={x: x_in, d:y_in})
        print(out[0], out[1], out[2])





