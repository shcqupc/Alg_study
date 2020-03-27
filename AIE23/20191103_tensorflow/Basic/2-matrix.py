import tensorflow as tf
import numpy as np
from datetime import datetime

print(('Your TensorFlow version: {0}').format(tf.__version__))
# if(tf.executing_eagerly()):
#     print('Eager execution is enabled (running operations immediately)\n')
#     print(('Turn eager execution off by running: \n{0}\n{1}').format('' \
#         'from tensorflow.python.framework.ops import disable_eager_execution', \
#         'disable_eager_execution()'))
# else:
#     print('You are not running eager execution. TensorFlow version >= 2.0.0' \
#           'has eager execution enabled by default.')
#     print(('Turn on eager execution by running: \n\n{0}\n\nOr upgrade '\
#            'your tensorflow version by running:\n\n{1}').format(
#            'tf.compat.v1.enable_eager_execution()',
#            '!pip install --upgrade tensorflow\n' \
#            '!pip install --upgrade tensorflow-gpu'))

# print(('Is your GPU available for use?\n{0}').format(
#     'Yes, your GPU is available: True' if tf.test.is_gpu_available() == True else 'No, your GPU is NOT available: False'
# ))
#
# print(('\nYour devices that are available:\n{0}').format(
#     [device.name for device in tf.config.experimental.list_physical_devices()]
# ))

a1 = tf.constant(np.ones([4, 4]) * 2, dtype=tf.float64, name="a1")
a2 = tf.Variable(np.ones([4, 4]), dtype=tf.float64, name="a2")
a1_X_a2 = tf.multiply(a1, a2, name='a1_X_a2')
print("a1_X_a2\n", a1_X_a2)
print('\n------------------------------------')
a3 = tf.zeros([4, 4], dtype=tf.float64, name="a3")
a4 = tf.ones([4, 4], dtype=tf.float64, name="a3")
d3_by_d4 = a3 * a4
print("d3_by_d4\n", d3_by_d4)
print('\n--------------matmul----------------------')
def matmul():
    a5 = tf.reshape(a3, [8, 2], name="a5")
    a6 = tf.reshape(a4, [2, 8], name="a6")
    a5_dot_a6 = tf.matmul(a5, a6, name='a5_dot_a6').numpy()
    print("a5_dot_a6\n", a5_dot_a6)
# matmul()
print('\n--------------tensordot----------------------')
def tensordot():
    a7 = tf.random.uniform((2, 3), maxval=10, dtype=tf.int64, name="a7")
    a8 = tf.random.uniform((1, 3), maxval=10, dtype=tf.int64, name="a8")
    print(a7,a8,sep="\n")
    a7_dot_a8 = tf.tensordot(a7,a8,name="a7_dot_a8",axes=0).numpy()
    print("a7_dot_a8\n", a7_dot_a8)
    print(a7_dot_a8.shape)
# tensordot()
print('\n--------------concat----------------------')
a9 = tf.random.uniform([3,3], maxval=10, dtype=tf.int64, name="a9")
a10 = tf.random.uniform([2,3], maxval=5, dtype=tf.int64, name="a10")
a11 = tf.random.uniform([3,2], maxval=5, dtype=tf.int64, name="a11")
print(np.shape(a9))
a9_c_a10 = tf.concat([a9,a10],axis=0).numpy()
print(a9_c_a10)
a9_c_a11 = tf.concat([a9,a11],axis=1).numpy()
print(a9_c_a11)
# stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = 'logs/%s' % stamp
# writer = tf.summary.create_file_writer(logdir)
# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
# tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.


# with writer.as_default():
#     tf.summary.trace_export(
#         name="tensordot",
#         step=0,
#         profiler_outdir=logdir)

# tf.summary.FileWriter("23-matrix", graph=graph)
# sess = tf.Session(graph=graph)
# print(sess.run(a1_dot_a2))
