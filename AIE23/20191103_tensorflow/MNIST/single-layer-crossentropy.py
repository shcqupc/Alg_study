#by cangye@hotmail.com
#引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#获取数据
mnist = input_data.read_data_sets("data/", one_hot=True)
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='data/mnist.npz')
#构建网络模型
#x，label分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
#构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#在使用交叉熵作为loss函数时，最后一层不加入激活函数
logits = tf.matmul(x, W) + b
#首先转换为概率分布
prob = tf.nn.softmax(logits)
#使用交叉熵作为损失函数
loss = tf.reduce_mean(tf.reduce_sum(- label * tf.log(prob), axis=1))
# 可以使用 
#loss = tf.losses.softmax_cross_entropy(label, logits)
#用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#用于验证
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
#迭代过程
train_writer = tf.summary.FileWriter("mnist-logdir", sess.graph)
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))
