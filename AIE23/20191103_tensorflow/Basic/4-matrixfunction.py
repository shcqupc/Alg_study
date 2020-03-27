import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

A = tf.placeholder(dtype=tf.float64, shape=[2, 2])
b = tf.placeholder(dtype=tf.float64, shape=[2])
#矩阵函数的使用
A_pow = tf.sin(A)
A_relu = tf.nn.relu(A)

A_inverse = tf.matrix_inverse(A)

A_T = tf.transpose(A)
b_diag = tf.diag(b)
I = tf.eye(6)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print('\n-------------A_pow-----------------------')
print(sess.run(A_pow, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))

print(sess.run(tf.sin(tf.constant([[1, 2], [-1, 1]],dtype=tf.float64))))

print('\n------------------------------------')
print(sess.run(A_relu, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))

print(sess.run(A_inverse, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))

print(sess.run([b_diag, I], 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))