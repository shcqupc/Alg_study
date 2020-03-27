# 引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

# 获取数据
mnist = input_data.read_data_sets("data/", one_hot=True)

# 构建网络模型
# x，label分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])

##########################################################
# 方法1 单层神经网络
# 构建单层网络中的权值和偏置
# W = tf.get_variable("W", [784, 10])
# b = tf.Variable(tf.zeros([10]))
# 本例中sigmoid激活函数 约束输出在0-1之间
# y = tf.nn.sigmoid(tf.matmul(x, W) + b)
# y = tf.matmul(x, W) + b

##########################################################
# 方法2 多层神经网络
# 增加神经网络，增加模型复杂度
# 总的可训练参数个数为 784*64 + 64*64 + 64*10
net = tf.layers.dense(x, 64, activation=tf.nn.leaky_relu)  # W1[784,64]
net = tf.layers.dense(net, 64, activation=tf.nn.leaky_relu)  # W2[64,64]
y = tf.layers.dense(net, 10, activation=None)  # W3[64,10]
print(np.shape(x), np.shape(label), np.shape(y))

##将输出y转换为概率
q = tf.nn.softmax(y)
##交叉熵作为损失函数
out = -label * tf.log(q)
out = tf.reduce_sum(out, axis=1)
loss = tf.reduce_mean(out)

# 用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 用于验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  ## 类型转换

# 定义会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())

## 指定一个文件用来保存训练过程的参数分布图
# train_writer = tf.summary.FileWriter("mnist-logdir", sess.graph)

## 读取模型
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('23-mnist'))

#########################迭代过程#############################
for itr in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 100 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))
## 存储模型
# saver.save(sess, "23-mnist/new", global_step=itr)

################################绘图过程################################################
import matplotlib.pyplot as plt
import matplotlib.style as sty

sty.use('fivethirtyeight')
# 获取W取值
W = sess.run(W.value())
# 绘图过程
fig = plt.figure()
ax = fig.add_subplot(221)
ax.matshow(np.reshape(W[:, 1], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(222)
ax.matshow(np.reshape(W[:, 2], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(223)
ax.matshow(np.reshape(W[:, 3], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(224)
ax.matshow(np.reshape(W[:, 4], [28, 28]), cmap=plt.get_cmap("Purples"))
plt.show()

