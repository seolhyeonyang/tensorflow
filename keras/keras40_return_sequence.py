import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


#* LSTM ( Long Short-Term Memory )
# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4,], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_predict = np.array([50, 60, 70])

print(x.shape, y.shape)     # (13, 3) (13,)

#x = x.reshape(13, 3, 1)
x = x.reshape(x.shape[0], x.shape[1], 1)

#x_predict = x_predict.reshape(1, 3, 1)
x_predict = x_predict.reshape(1, x_predict.shape[0], 1)

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 7)                 504
_________________________________________________________________
dense (Dense)                (None, 4)                 32
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 5
=================================================================
Total params: 1,021
Trainable params: 1,021
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
import time
es = EarlyStopping(monitor='loss', patience=50,mode='min', verbose=1)

start_time = time. time()
model.fit(x, y, epochs=10000, batch_size=1, verbose=2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가, 예측
results = model.predict(x_predict)
print('걸린시간 : ', end_time)
print(results)


'''
걸린시간 :  18.76767349243164
[[80.319244]]

LSTM 2번
걸린시간 :  32.90895628929138
[[82.97339]]

Dropout
걸린시간 :  12.617850542068481
[[90.008736]]
'''