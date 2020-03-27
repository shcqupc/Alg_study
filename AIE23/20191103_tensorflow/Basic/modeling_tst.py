import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
x = tf.placeholder(tf.float32, [None, 1])
d = tf.placeholder(tf.float32, [None, 1])
# get variables
w = tf.get_variable("w", [1, 1])
b = tf.get_variable("b", [1])
# define model
y = tf.matmul(x, w) + b
# define loss function
loss = (y - d) * 2
loss = tf.reduce_mean(loss)
# define optimizer and learning rate
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)
'''
# calculate gradients for w/b
optimizer.compute_gradients(loss, [w, b])
# perform
train_step = optimizer.apply_gradients(grads)
'''
file = np.load("homework.npz")
data_x = file['X']
data_d = file['d']
# input samples and train
# define session
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
for step in range(200):
    st, ls = sess.run([train_step, loss], feed_dict={x: data_x, d: data_d})
    print(ls)
out = sess.run(y, feed_dict={x: data_x})
import matplotlib.pyplot as plt

plt.scatter(x[:, 0], d[:, 0])
plt.scatter(x[:, 0], y[:, 0])
plt.show()
