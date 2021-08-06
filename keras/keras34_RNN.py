import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout


# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4,], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)     # (4, 3) (4,)

x = x.reshape(4, 3, 1)
#! (batch, timesteps, feature) 
#! feature: 몇 개씩 자르는지
# 1, 2, 3, 4, 5, 6, 7, 8 feature 2이면 1, 2  합쳐서 계산하고 3, 4 같이 계산 5, 6 계산해서 7, 8을 예측

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
#model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================       param
simple_rnn (SimpleRNN)       (None, 25)                675              units * (units + input_dim + 1(바이어스))
_________________________________________________________________       25 * (25 + 1 + 1)
dropout (Dropout)            (None, 25)                0
_________________________________________________________________
dense (Dense)                (None, 13)                338
_________________________________________________________________
dropout_1 (Dropout)          (None, 13)                0
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 84
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 7
=================================================================
Total params: 1,104
Trainable params: 1,104
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50,mode='min', verbose=1)

model.fit(x, y, epochs=1000, batch_size=1, verbose=2, callbacks=[es])

# 4. 평가, 예측
x_input = np.array([5, 6, 7]).reshape(1,3,1)
results = model.predict(x_input)
print(results)      # [[7.999995]]