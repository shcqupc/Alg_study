#参考文章https://zhuanlan.zhihu.com/p/28446996
import tensorflow.compat.v1 as tf
with tf.variable_scope("first-nn-layer"):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    W1 = tf.Variable(tf.zeros([784, 10]), name="W")
print(W.name)
print(W1.name)