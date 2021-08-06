import numpy as np
from tensorflow.python.keras.saving.model_config import model_from_config

#1. 데이터
x = np.array([range(100), range(301,401), range(1,101),
            range(100), range(401,501)])
x = np.transpose(x)

print(x.shape)      # (100, 5)

y = np.array([range(711,811), range(101,201)])
y = np.transpose(y)

print(y.shape)      # (100, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model       #Model 함수형 모델에 사용
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs = input1, outputs=output1)
'''
각 레이어 마다 정의 해줘야 한다. 
상위레이어를 하위레이어에 적용해 준다.
model은 마지막에 input과 output 을 지정해 줘서 정의 한다.
여러 모델을 합치는 데 편하다.

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
=================================================================
Total params: 106
Trainable params: 106
Non-trainable params: 0
_________________________________________________________________

함수형은 input 부터 명시

'''

'''
순차형
model = Sequential()
model.add(Dense(3, input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

단일 모델로는 편한데
여러 모델를 합치는데 힘들다.

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
=================================================================
Total params: 106
Trainable params: 106
Non-trainable params: 0
_________________________________________________________________
'''

model.summary()


#3. 컴파일, 훈련
#4. 평가, 예측 