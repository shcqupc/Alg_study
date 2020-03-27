# by cangye@hotmail.com
# TensorFlow入门实例
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.decomposition import PCA


def variable_summaries(var, name="layer"):
    with tf.variable_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

data = pd.read_csv("data/iris.data.csv")
c_name = set(data.name.values)
print(c_name)
iris_label = np.zeros([len(data.name.values),len(c_name)])
iris_data = data.values[:, :-1]
pca = PCA(n_components=2)
iris_data = pca.fit(iris_data).transform(iris_data)
iris_data = iris_data-np.mean(iris_data, axis=0)
iris_data = iris_data/np.max(iris_data, axis=0)
train_data=[]
train_data_label=[]
test_data=[]
test_data_label=[]
for idx, itr_name in enumerate(c_name):
    datas_t = iris_data[data.name.values==itr_name, :]
    labels_t = np.zeros([len(datas_t), len(c_name)])
    labels_t[:, idx] = 1
    train_data.append(datas_t[:30])
    train_data_label.append(labels_t[:30])
    test_data.append(datas_t[30:])
    test_data_label.append(labels_t[30:])
train_data = np.concatenate(train_data)
train_data_label = np.concatenate(train_data_label)
test_data = np.concatenate(test_data)
test_data_label = np.concatenate(test_data_label)
x = tf.placeholder(tf.float32, [None, 2], name="input_x")
label = tf.placeholder(tf.float32, [None, 3], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = slim.fully_connected(x, 4, activation_fn=tf.nn.sigmoid, 
                              scope='full1', reuse=False)
net = tf.contrib.layers.batch_norm(net)
net = slim.fully_connected(net, 8, activation_fn=tf.nn.sigmoid, 
                              scope='full2', reuse=False)
net = tf.contrib.layers.batch_norm(net)
net = slim.fully_connected(net, 8, activation_fn=tf.nn.sigmoid, 
                              scope='full3', reuse=False)
net = tf.contrib.layers.batch_norm(net)
net = slim.fully_connected(net, 4, activation_fn=tf.nn.sigmoid, 
                              scope='full4', reuse=False)
net = tf.contrib.layers.batch_norm(net)
y = slim.fully_connected(net, 3, activation_fn=tf.nn.sigmoid, 
                              scope='full5', reuse=False)
loss = tf.reduce_mean(tf.square(y-label))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(0.5)
var_list_w = [var for var in tf.trainable_variables() if "w" in var.name]
var_list_b = [var for var in tf.trainable_variables() if "b" in var.name]
gradient_w = optimizer.compute_gradients(loss, var_list=var_list_w)
gradient_b = optimizer.compute_gradients(loss, var_list=var_list_b)
for idx, itr_g in enumerate(gradient_w):
    variable_summaries(itr_g, "layer%d-w-grad"%idx)
for idx, itr_g in enumerate(gradient_b):
    variable_summaries(itr_g, "layer%d-b-grad"%idx)
for idx, itr_g in enumerate(var_list_w):
    variable_summaries(itr_g, "layer%d-w"%idx)
train_step = optimizer.apply_gradients(gradient_w+gradient_b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter("logdir-bn", sess.graph)
merged = tf.summary.merge_all()
for itr in range(600):
    sess.run(train_step, feed_dict={x: train_data, label: train_data_label})
    if itr % 30 == 0:
        acc1 = sess.run(accuracy, feed_dict={x: train_data,
                                        label: train_data_label})
        acc2 = sess.run(accuracy, feed_dict={x: test_data,
                                        label: test_data_label})
        print("step:{:6d}  train:{:.3f} test:{:.3f}".format(itr, acc1, acc2))
        summary = sess.run(merged, 
                           feed_dict={x: train_data,
                                        label: train_data_label})
        train_writer.add_summary(summary, itr)

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

mpl.style.use('fivethirtyeight')
xx = yy = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(xx, yy)
sp = np.shape(X)
xl = np.reshape(X, [-1, 1])
yl = np.reshape(Y, [-1, 1])
gridxy = np.concatenate([xl, yl], axis=1)
print(np.shape(gridxy))
zl = sess.run(y, feed_dict={x: gridxy})
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
print(len(iris_data))
surface=ax.plot_surface(X, Y, np.reshape(zl[:,0],sp), alpha=0.2)
surface=ax.plot_surface(X, Y, np.reshape(zl[:,1],sp), alpha=0.2)
surface=ax.plot_surface(X, Y, np.reshape(zl[:,2],sp), alpha=0.2)
ax.scatter(iris_data[:50,0], iris_data[:50,1], color="#990000", s=60)
ax.scatter(iris_data[50:100,0], iris_data[50:100,1], color="#009900", s=60)
ax.scatter(iris_data[100:,0], iris_data[100:,1], color="#000099", s=60)
plt.show()