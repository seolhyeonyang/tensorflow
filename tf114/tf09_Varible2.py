import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b 

# TODO tf09_1 방식으로 3가지를 출력하시오!

sess = tf.Session()
sess.run(tf.global_variables_initializer())
h1 = sess.run(hypothesis)
print('1번째 : ', h1)
# 1번째 :  [1.3       1.6       1.9000001]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
h2 = hypothesis.eval()
print('2번째 : ', h2)
# 2번째 :  [1.3       1.6       1.9000001]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
h3 = hypothesis.eval(session = sess)
print('3번째 : ', h3)
# 3번째 :  [1.3       1.6       1.9000001]

sess.close()