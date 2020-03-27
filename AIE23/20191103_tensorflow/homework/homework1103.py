import numpy as np
import tensorflow as tf

file = np.load("homework1103.npz")
X = file["X"]
label = file["d"]
print("X:" + str(np.shape(X)), "label:" + str(np.shape(label)))
# label_one_hot = label[:,]




def make_one_hot(data1):
    return (np.arange(2) == data1[:, None]).astype(np.integer)


label_one_hot = make_one_hot(label)
# print(tf.concat([np.array(X[:5, ]), np.array(label_one_hot[:5])], axis=1).numpy())
print(tf.concat([X[:5, ], label_one_hot[:5]], axis=1).numpy())

import tensorflow.contrib.slim as slim
x = tf.placeholder(tf.float32, [None, 2], name="input_x")
d = tf.placeholder(tf.float32, [None, 2], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu,
                              scope='full1', reuse=False)
net = slim.fully_connected(net, 4, activation_fn=tf.nn.relu,
                              scope='full4', reuse=False)
y = slim.fully_connected(net, 2, activation_fn=None,
                              scope='full5', reuse=False)
loss = tf.reduce_mean(tf.square(y-d))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.1)
gradient = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
train_step = optimizer.apply_gradients(gradient)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for itr in range(500):
    idx = np.random.randint(0, 2000, 20)
    inx = X[idx]
    ind = label_one_hot[idx]
    sess.run(train_step, feed_dict={x:inx, d:ind})
    if itr%10 == 0:
        acc = sess.run(accuracy, feed_dict={x:X, d:label_one_hot})
        print("step:{}  accuarcy:{}".format(itr, acc))