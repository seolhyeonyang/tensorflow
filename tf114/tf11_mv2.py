import tensorflow as tf
tf.set_random_seed(66)


x_data = [[73, 51, 65],
        [92, 98, 11],
        [89, 31, 33],
        [99, 33, 100],
        [17, 66, 79]]

y_data = [[152], [185], [180], [205], [142]]
# x_data.shape = (5,3) / y_data.shape = (5, 1)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
#! 행은 나중에 추가 될 수 있어서 None

w = tf.Variable(tf.random.normal([3,1]), name='weight')
#! input shape 맞춰서 shape 를 맞춰야 한다. -> [3,1]
#^ 행렬 곱하기는 (5,3) * (3,1) = (5,1)  -> 앞 행렬 '행' 과 뒤 행렬 '열'이 같아야 가능 / 없어져 계산된 행렬은 (5, 1) 된다.
#^ y_data shape 와 같게 만들어 준다.
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b
#! tf.matmul(x, w) 행렬 연산 해주는 것

cost = tf.reduce_mean(tf.square(hypothesis - y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                feed_dict={x:x_data, y:y_data})
        if epochs % 10 == 0:
                print(epochs, 'cost : ', cost_val, '\n', hy_val)


sess.close()