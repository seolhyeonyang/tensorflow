from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from keras.utils import to_categorical
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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random.normal([28*28, 256]), name='weight1')
b1 = tf.Variable(tf.random.normal([256]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
layer2 = tf.nn.dropout(layer1, keep_prob=0.3)

w2 = tf.Variable(tf.random.normal([256, 128]), name='weight2')
b2 = tf.Variable(tf.random.normal([128]), name='bias2')
layer3 = tf.nn.relu(tf.matmul(layer2, w2) + b2)
layer4 = tf.nn.dropout(layer3, keep_prob=0.3)

w3 = tf.Variable(tf.random.normal([128, 64]), name='weight3')
b3 = tf.Variable(tf.random.normal([64]), name='bias3')
layer5 = tf.nn.relu(tf.matmul(layer4, w3) + b3)
layer6 = tf.nn.dropout(layer5, keep_prob=0.3)

w = tf.Variable(tf.random.normal([64, 10]), name='weight')
b = tf.Variable(tf.random.normal([10]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(layer6, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
start_time = time.time()

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict={x:x_train, y:y_train})
    if epochs % 1000 == 0:
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

sess.close()