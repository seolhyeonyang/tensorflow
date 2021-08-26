from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
import time
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)      # (60000,) (10000,)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

scalar = MinMaxScaler()
# scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. 모델
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])


w = tf.Variable(tf.random.normal([28*28,10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)


cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.017)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000095)

train = optimizer.minimize(cost)

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())

    # 3. 훈련
    start_time = time.time()

    for epochs in range(501):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                        feed_dict={x:x_train, y:y_train})
        if epochs % 50 == 0:
            print(epochs, 'cost : ', cost_val, '\n', hy_val)

    end_time = time.time() - start_time

    # 4. 평가 예측
    results = sess.run(hypothesis, feed_dict={x:x_test})
    y_pred = sess.run(tf.argmax(results, 1))
    print('예측값 : ', y_pred)

    y_test = np.argmax(y_test, axis=1)

    accuray = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_test), dtype=tf.float32))

    print('time : ', end_time)
    print('acc : ', sess.run(accuray))
    # print('acc_score : ', accuracy_score(y_test, y_pred))
