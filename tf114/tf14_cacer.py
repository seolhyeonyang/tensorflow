from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy  as np
import tensorflow as tf
tf.set_random_seed(66)


datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

y_data = y_data.reshape(-1,1)

# print(x_data.shape, y_data.shape)     # (569, 30) (569, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=78)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

# accuracy_score로 할것

w = tf.Variable(tf.random.normal([30,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, 'cost : ', cost_val, '\n', hy_val)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuray = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

pred, acc = sess.run([predicted, accuray], feed_dict={x:x_test, y:y_test})
print('=======================================')
print('예측값 : \n', hy_val,
    '\n예측 결과값 : \n', pred,
    '\nacc : ', np.round(acc,5))

sess.close()

acc_score = accuracy_score(y_test, pred)
print('accuracy_score : ', np.round(acc_score,5))

# acc :  0.37258
# accuracy_score :  0.37258