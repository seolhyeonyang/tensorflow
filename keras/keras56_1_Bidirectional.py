import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional


#* LSTM ( Long Short-Term Memory )
# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4,], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)     # (4, 3) (4,)

x = x.reshape(4, 3, 1)


# 2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(Bidirectional(LSTM(units=10, activation='relu', input_shape=(3,1))))
#! bidirectional 첫번째 레이어에 쓰기 힘들다.
#! input_shape가 거꾸로 가야하는데 그것 때문에 오류가 난다.
#^ 연산량은 안쓴게의 2배이다.

model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
bidirectional (Bidirectional (None, 20)                1680
_________________________________________________________________
dense (Dense)                (None, 5)                 105
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================
Total params: 2,271
Trainable params: 2,271
Non-trainable params: 0
_________________________________________________________________
'''

exit()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50,mode='min', verbose=1)

model.fit(x, y, epochs=10000, batch_size=1, verbose=2, callbacks=[es])

# 4. 평가, 예측
x_input = np.array([5, 6, 7]).reshape(1,3,1)
results = model.predict(x_input)
print(results)      # [[7.9990306]]