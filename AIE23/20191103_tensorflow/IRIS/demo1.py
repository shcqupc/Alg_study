# by cangye@hotmail.com
# TensorFlow入门实例

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

#获取数据
data = pd.read_csv("../data/iris.data.csv")
#获取数据类
c_name = set(data.name.values)
print(c_name)
#获取标签
iris_label = np.zeros([len(data.name.values),len(c_name)])
#获取数据
iris_data = data.values[:, :-1]
#去均值、归一化
iris_data = iris_data-np.mean(iris_data, axis=0)
iris_data = iris_data/np.max(iris_data, axis=0)
print(iris_data)
len_of_data = []
for idx, itr_name in enumerate(c_name):
    len_of_data.append(len(iris_label[data.name.values==itr_name]))
    iris_label[data.name.values==itr_name, idx] = 1
print(len_of_data)
x = tf.placeholder(tf.float32, [None, 4], name="input_x")
label = tf.placeholder(tf.float32, [None, 3], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
#net = tf.layers.dense(x, 28, activation=tf.nn.relu)
y = tf.layers.dense(x, 3, activation=tf.nn.sigmoid)
loss = tf.reduce_mean(tf.square(y-label))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(100):
    sess.run(train_step, feed_dict={x: iris_data[:100], label: iris_label[:100]})
    if itr % 30 == 0:
        acc = sess.run(accuracy, feed_dict={x: iris_data[:100],
                                        label: iris_label[:100]})
        print("step:{:6d}  accuracy:{:.3f}".format(itr, acc))
