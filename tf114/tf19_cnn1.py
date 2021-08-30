import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
tf.set_random_seed(66)


#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
tatal_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

#2. 모델구성
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])
#! shape=[kernel_size, input, output]
# w2 = tf.Variable(tf.random_normal([3, 3, 1, 32]), dtype=tf.float32)
# w3 = tf.Variable([1], dtype=tf.float32)
#! Variable 과 get_variable 차이점
#^ Variable은 언제나 새로운 객체를 만든다. (초기값 줘야 한다.)
#^ get_variable은 재사용이 가능하다. (shape를 줘야 한다.)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(np.min(sess.run(w1)))
# print(np.max(sess.run(w1)))
# print(np.mean(sess.run(w1)))
# print(np.median(sess.run(w1)))

# print(sess.run(w1))
# print(w1)
# <tf.Variable 'w1:0' shape=(3, 3, 1, 32) dtype=float32_ref>

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
#! x = input
#! w1(shape=[3, 3, 1, 32])에서 (3, 3)=kernel_size, (1)=x 마지막과 맞춰줌, (32)=output
#! strides=[1, 1, 1, 1] -> 가운데 2개가 중요 양 끝은 차원 맞춰준것
#^ strides은 몇칸씩 뛸 것인가 대부분 1을 준다. 2는 maxpool 할때 사용
#^ padding은 대문자만 가능 VALID일때 output은 (input - kernel +1) 이다.

# model = Sequential()
# model.add(Conv2D(filter=32, kernel_size=(3, 3), strides=1,
#                 padding='same', input_shape=(28, 28, 1)))
#! tensorflow2 에 구현 하면 이렇게 구현 input_shape=(low, cols, chaneel)