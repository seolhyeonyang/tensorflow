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

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1, 2, 3], y_train:[3, 5, 7]})
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val)

'''
[4]
[5, 6]
'''
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis = x_test * W_val + b_val

print(sess.run(hypothesis, feed_dict={x_test:[4]}))
print(sess.run(hypothesis, feed_dict={x_test:[5, 6]}))
print(sess.run(hypothesis, feed_dict={x_test:[7, 8, 9]}))
