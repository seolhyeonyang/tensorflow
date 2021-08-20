import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
#? kernelregulaizer => kernel은 weights이다. weight를 규제하는 것

# print(model.weights)
# print('='*200)
# print(model.trainable_weights)
#! model.weights = model.trainable_weights 같다.

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.7836573 , -0.8753456 ,  0.24897921]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
#! dense/kernel:0 첫번째 연산 / shape=(1, 3) -> 인풋 1개, 아웃풋 3개
#! bias / shape=(3,) 아웃풋이 3개라 bias도 3개 필요(bias 디폴트가 0)
#. y = w1 * x + w2 * x + w3 * x

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.32853162, -0.9932635 ],
        [ 0.22664535,  0.08078933],
        [-0.4614355 ,  0.6275046 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.60339415],
        [ 0.08079219]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
========================================================================================================================================================================================================
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-0.7836573 , -0.8753456 ,  0.24897921]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.32853162, -0.9932635 ],
        [ 0.22664535,  0.08078933],
        [-0.4614355 ,  0.6275046 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[-0.60339415],
        [ 0.08079219]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

# model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
'''

print(len(model.weights))
print(len(model.trainable_weights))
'''
6
6
#! layer마다 weight 와 bias해서 3(w + b) - 지금은 3개의 레이어가 있어서 한 층에 한 개의 w, b
'''