import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.compat.v1.set_random_seed(77)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

#! sess 표현 방식
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print('aaa : ', aaa)
# aaa :  [1.014144]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('bbb : ', bbb)
# bbb :  [1.014144]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print('ccc : ', ccc)
# ccc :  [1.014144]

sess.close()
