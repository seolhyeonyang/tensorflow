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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary()

'''
summary -> 모델의 자세한 내용을 알려준다.

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

Model -> 어떤 모델인지 명시

layer(type) -> model의 연산을 어떤것을 사용했는지

Param -> (input +1) * output
        +1 은 바이어스(b)를 뜻한다.

Total params -> 전체 연산 횟수
Trainable params -> 훈련 시킨 연산 횟수
'''

#3. 컴파일, 훈련
#4. 평가, 예측 