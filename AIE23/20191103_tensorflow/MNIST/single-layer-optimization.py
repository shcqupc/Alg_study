#by cangye@hotmail.com
#引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#定义全链接层函数
def full_layer(input_tensor, out_dim, name='full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return tf.nn.sigmoid(out)

def model(net, out_dim):
    net=full_layer(net, out_dim, "full_layer1")
    return net

#定义输入
with tf.variable_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
#引入模型
y = model(x, 10)
#定义损失函数
loss = tf.reduce_mean(tf.square(y-label))
#用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#用于验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("mnist-logdir", sess.graph)
#迭代过程
mnist = input_data.read_data_sets("data/", one_hot=True)
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))

#这下面的部分用于绘图
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.style.use('fivethirtyeight')
#获取W取值
W = sess.run(W.value())
#绘图过程
fig = plt.figure()
ax = fig.add_subplot(221)
ax.matshow(np.reshape(W[:,1],[28,28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(222)
ax.matshow(np.reshape(W[:,2],[28,28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(223)
ax.matshow(np.reshape(W[:,3],[28,28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(224)
ax.matshow(np.reshape(W[:,4],[28,28]), cmap=plt.get_cmap("Purples"))
plt.show()