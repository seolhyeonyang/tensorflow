# y = wx + b
# w, b  변하는 값이니 변수
# x, y 입력받는 값이니 placeholder

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(66)


x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

hypothesis = x_train * W +b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))      # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# sess = Session()
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(W), sess.run(b))
