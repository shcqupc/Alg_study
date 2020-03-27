import tensorflow.compat.v1 as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# 需要从外界接收样本：placeholder
x = tf.placeholder(tf.float32, [None, 1])
d = tf.placeholder(tf.float32, [None, 1])

# 需要定义可训练参数：Variable

# w = tf.get_variable("w", [1, 1])
# b = tf.get_variable("b", [1])
# 需要定义模型: y=xw+b
# y = tf.matmul(x, w) + b


w1 = tf.get_variable("w1", [1, 1000])
b1 = tf.get_variable("b1", [1000])
w2 = tf.get_variable("w2", [1000, 1])
b2 = tf.get_variable("b2", [1])
# 需要定义模型: y=xw+b
################################
# 一个全连接层
# h = tf.matmul(x, w1) + b1
# h = tf.nn.relu(h) #激活函数
################################
# y = tf.matmul(h, w2) + b2

# 将全连接层封装成高层次API
h = tf.layers.dense(
    x,
    1000,
    activation=tf.nn.relu)
y = tf.layers.dense(
    h,
    1,
    activation=None)

# 定义优化函数：loss函数
loss = (y - d) ** 2
loss = tf.reduce_mean(loss)
# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_step = optimizer.minimize(loss)

# 输入样本进行训练
# 定义会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())

## 读取训练数据
import numpy as np

# file = np.load("homework.npz")
# data_x = file['X']
# data_d = file['d']
data_x = np.random.random([3000, 1]) * 6
# data_x = np.random.normal(0,2, [3000,1])
data_d = np.sin(data_x)

for step in range(1000):
    ind = np.random.randint(0, 1000, [32])
    inx = data_x[ind]
    ind = data_d[ind]
    st, ls = sess.run(
        [train_step, loss],
        feed_dict={
            x: data_x,
            d: data_d
        }
    )
    print(ls)

pred_y = sess.run(y, feed_dict={x: data_x})
import matplotlib.pyplot as plt

plt.figure(figsize=[4, 3])
plt.scatter(data_x[:, 0], data_d[:, 0], c="y", lw=0.2)
plt.scatter(data_x[:, 0], pred_y[:, 0], c="r", lw=0.2)
plt.show()
