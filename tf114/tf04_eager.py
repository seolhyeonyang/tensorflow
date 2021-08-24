import tensorflow as tf
print(tf.__version__)

# print('hello world')

tf.compat.v1.disable_eager_execution()
#! v2에서는 Session 사용불가 / 사용가능하게 해주는 것

hello = tf.constant('Hello World')

print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
#! 버전을 맞춰 줘야함
print(sess.run(hello))
#. b'Hello World'