import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.tensor_shape import scalar


datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=78)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

scalar = MinMaxScaler()
# scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)
print(y_train.shape, y_test.shape)      # (120, 3) (30, 3)

# print(x_train[:5])

x = tf.placeholder(tf.float32, shape=(None, 4))
y = tf.placeholder(tf.float32, shape=(None, 3))

w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x , w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(2001):
        cost_val, _ = sess.run([loss, optimizer], feed_dict={x:x_train, y:y_train})
        if epochs % 1000 == 0:
            print(epochs, 'cost : ', cost_val)

    results = sess.run(hypothesis, feed_dict={x:x_test})

    accuray = tf.reduce_mean(tf.cast(tf.equal(results, y_test), dtype=tf.float32))

    print(results, sess.run(tf.argmax(results, 1)))
    print('acc : ', sess.run(accuray))