#参考文章https://zhuanlan.zhihu.com/p/28446996
import tensorflow as tf
with tf.variable_scope("first-nn-layer") as scope:
    W = tf.get_variable("W", [784, 10])
    b = tf.get_variable("b", [10])
    scope.reuse_variables()
    W1 = tf.get_variable("W", shape=[784, 10])
print(W.name)
print(W1.name)

with tf.variable_scope("first-nn-layer") as scope:
    W = tf.get_variable("W", [784, 10])
    b = tf.get_variable("b", [10])
with tf.variable_scope("second-nn-layer") as scope:
    W = tf.get_variable("W", [784, 10])
    b = tf.get_variable("b", [10])
with tf.variable_scope("second-nn-layer", reuse=True):
    W3 = tf.get_variable("W", [784, 10])
    b3 = tf.get_variable("b", [10])
print(W.name)
print(W3.name)