#by cangye@hotmail.com
#引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#获取数据
mnist = input_data.read_data_sets("data/", one_hot=True)
#构建网络模型
#x，label分别为图形数据和标签数据

x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
#构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
tf.summary.histogram('W', W)
b = tf.Variable(tf.zeros([10]))
#本例中为sigmoid激活函数
y = tf.nn.sigmoid(tf.matmul(x, W) + b)
#定义损失函数为欧氏距离
loss = tf.reduce_mean(tf.square(y-label))
tf.summary.scalar('loss', loss)
#用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#用于验证
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float3W2))
#定义会话
sess = tf.Session()
#初始化所有变量
sess.run(tf.global_variables_initializer())
#定义log，用于tensorboard观察
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("mnist-logdir", sess.graph)
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))
        summary = sess.run(merged, 
                           feed_dict={x: batch_xs,
                                        label: batch_ys})
        train_writer.add_summary(summary, itr)