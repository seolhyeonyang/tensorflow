# y = wx + b
# w, b  변하는 값이니 변수
# x, y 입력받는 값이니 placeholder

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(77)
#^ radom_state 와 같은것

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable([1], dtype=tf.float32)
# b = tf.Variable([1], dtype=tf.float32)

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.172)
train = optimizer.minimize(loss)

sess = Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    # if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
    print(step, loss_val, W_val, b_val)

'''
[4]
[5, 6]
'''
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis = x_test * W_val + b_val

print(sess.run(hypothesis, feed_dict={x_test:[4]}))

'''
learning_rate=0.15
95 1.6416621e-05 [2.0045362] [0.98968834]
96 1.5253646e-05 [2.0043726] [0.99006015]
97 1.4174061e-05 [2.0042148] [0.9904185]
98 1.3170014e-05 [2.0040631] [0.9907641]
99 1.2237226e-05 [2.0039163] [0.991097]
[9.006762]

learning_rate=0.17
95 6.358629e-06 [2.002774] [0.9936048]
96 5.8471537e-06 [2.0027215] [0.99389285]
97 5.378224e-06 [2.0025563] [0.9941187]
98 4.947369e-06 [2.0024996] [0.99438006]
99 4.5507127e-06 [2.002355] [0.9945911]
[9.004011]

learning_rate=0.173
95 1.5325713e-05 [2.0014918] [0.99357134]
96 1.3352353e-05 [2.0035317] [0.9947633]
97 1.1653159e-05 [2.001453] [0.9941312]
98 1.0187084e-05 [2.003168] [0.9951564]
99 8.918821e-06 [2.0014045] [0.99464]
[9.000258]
'''