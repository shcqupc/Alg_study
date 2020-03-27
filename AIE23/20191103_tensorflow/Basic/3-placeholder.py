import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
a1 = tf.constant(np.ones([4, 4])*2)
a2 = tf.constant(np.ones([4, 4]))
b1 = tf.Variable(a1)
b2 = tf.Variable(np.ones([4, 4]))
#定义placeholder
c2 = tf.placeholder(dtype=tf.float64, shape=[4, 4])

a1_elementwise_a2 = a1*a2
a1_dot_a2 = tf.matmul(a1, a2)

b1_elementwise_b2 = b1 * b2
b1_dot_b2 = tf.matmul(b1, b2)

c2_dot_b2 = tf.matmul(c2, b2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(c2_dot_b2, feed_dict={c2: np.zeros([4, 4])}))
