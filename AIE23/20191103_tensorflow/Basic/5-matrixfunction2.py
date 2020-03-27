import tensorflow as tf
import numpy as np

A = tf.placeholder(dtype=tf.float32, shape=[2, 2])
B = tf.placeholder(dtype=tf.float64, shape=[2, 2])
b = tf.placeholder(dtype=tf.float64, shape=[2])

A = tf.cast(A, tf.float64)
#最常用到的矩阵乘法
A_dot_B = tf.matmul(A, B)

#矩阵连接操作，在大型神经网络中用的比较多
AA0 = tf.concat([A, A], axis=0)
AA1 = tf.concat([A, A], axis=1)
#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(sess.run(A_dot_B, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          B: [[1, 2], [-1, 1]]}))

print(sess.run(AA0, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          B: [[1, 2], [-1, 1]]}))

print(sess.run(AA1, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          B: [[1, 2], [-1, 1]]}))

