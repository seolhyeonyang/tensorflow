import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer()
#! tensorflow 변수는 반드시 초기화 해줘야 한다. (값 초기화가 이니라 연산초기화)

sess.run(init)
print(sess.run(x))
