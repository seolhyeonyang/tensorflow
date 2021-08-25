from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.set_random_seed(66)


datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target

y_data = y_data.reshape(-1,1)

# print(x_data.shape, y_data.shape)     # (442, 10) (442, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=77)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None,1])

# r2_score로 할것

w = tf.Variable(tf.random.normal([10,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.95)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(8001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                feed_dict={x:x_train, y:y_train})
        if epochs % 1000 == 0:
                print(epochs, 'cost : ', cost_val, '\n', hy_val)

predicted = sess.run(hypothesis, feed_dict={x:x_test})

r2 = r2_score(y_test, predicted)
print('r2 : ', r2)

sess.close()

# r2 :  0.5034040864968545