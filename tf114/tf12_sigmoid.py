import tensorflow as tf
tf.set_random_seed(66)


# 1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [4, 3], [6, 2]]   # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                     # (6, 1)


# 2. 모델
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
#! 값을 0 ~ 1 사이로 해준다.

# cost = tf.reduce_mean(tf.square(hypothesis - y))        # mse
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))   # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost : ', cost_val, '\n', hy_val)


# 4. 평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#! cast 0, 1로 나타낸다. -> hypothesis > 0.5이면 1 아니면 0

accuray = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
#! predicted 와 y가 같으면 1, 아니면 0으로 해서 평균을 낸다.

pred, acc = sess.run([predicted, accuray], feed_dict={x:x_data, y:y_data})
print('=======================================')
print('예측값 : \n', hy_val,
    '\n예측 결과값 : \n', pred,
    '\nacc : ', acc)

sess.close()