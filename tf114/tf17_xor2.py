import tensorflow as tf
tf.set_random_seed(66)

#! preceptron -> mlp

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]   # (4, 2)
y_data = [[0], [1], [1], [0]]               # (4, 1)


# 2. 모델
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

#* 히든레이어 1
w1 = tf.Variable(tf.random.normal([2,3]), name='weight1')
b1 = tf.Variable(tf.random.normal([3]), name='bias1')

h_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

#* 히든레이어 2
w2 = tf.Variable(tf.random.normal([3, 5]), name='weight2')
b2 = tf.Variable(tf.random.normal([5]), name='bias2')

h_layer2 = tf.sigmoid(tf.matmul(h_layer1, w2) + b2)

#* 아웃풋 레이어
w = tf.Variable(tf.random.normal([5,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')
#! 행렬 연산으로 연결해 주면 된다. mlp 구성 가능

hypothesis = tf.sigmoid(tf.matmul(h_layer2, w) + b)


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))   # binary_crossentropy

# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 1000 == 0:
        print(epochs, 'cost : ', cost_val, '\n', hy_val)


# 4. 평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuray = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

pred, acc = sess.run([predicted, accuray], feed_dict={x:x_data, y:y_data})
print('=======================================')
print('예측값 : \n', hy_val,
    '\n예측 결과값 : \n', pred,
    '\nacc : ', acc)

sess.close()


# 예측값 :
#  [[0.01292424]
#  [0.9765414 ]
#  [0.9787257 ]
#  [0.02431441]]

# 예측 결과값 :
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# acc :  1.0
