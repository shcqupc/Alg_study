import tensorflow as tf
import numpy as np
#定义常量
a1 = tf.constant(np.ones([4, 4])*2)
a2 = tf.constant(np.ones([4, 4]))
#定义变量
b1 = tf.Variable(a1)
b2 = tf.Variable(np.ones([4, 4]))
#定义乘法
a1_elementwise_a2 = a1*a2
#定义矩阵乘法
a1_dot_a2 = tf.matmul(a1, a2)

b1_elementwise_b2 = b1 * b2
b1_dot_b2 = tf.matmul(b1, b2)

a1b2 = tf.matmul(a1,b2)
print(a1_elementwise_a2)
print(a1_dot_a2)
print(b1_elementwise_b2)
print(b1_dot_b2)
print(a1b2)
#variable需要初始化
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# print(sess.run(a1_elementwise_a2))
# print(sess.run(a1_dot_a2))
# print(sess.run(b1_elementwise_b2))
# print(sess.run(b1_dot_b2))
