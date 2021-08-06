import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input, concatenate


# 1. 데이터
x1 = np.array([[1, 2, 3], [2, 3, 4,], [3, 4, 5], [4, 5, 6],
            [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
            [9, 10, 11], [10, 11, 12],
            [20, 30, 40], [30, 40, 50], [40, 50, 60]])

x2 = np.array([[10, 20, 30], [20, 30, 40,], [30, 40, 50], [40, 50, 60],
            [50, 60, 70], [60, 70, 80], [70, 80, 90], [80, 90, 100],
            [90, 100, 110], [100, 110, 120],
            [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = np.array([55, 65, 75])
x2_predict = np.array([65, 75, 85])

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

x1_predict = x1_predict.reshape(1, x1_predict.shape[0], 1)
x2_predict = x2_predict.reshape(1, x2_predict.shape[0], 1)

# 2. 모델 구성
# 2-1. 모델1
input1 = Input(shape=(3,1))
lstm1 = LSTM(32, activation='relu')(input1)
dense1 = Dense(16, activation='relu')(lstm1)
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(4, activation='relu')(dense2)
output1 = Dense(1)(dense3)

# 2-2. 모델2
input2 = Input(shape=(3,1))
lstm2 = LSTM(32, activation='relu')(input2)
dense11 = Dense(16, activation='relu')(lstm2)
dense12 = Dense(8, activation='relu')(dense11)
dense13 = Dense(4, activation='relu')(dense12)
output2 = Dense(1)(dense13)

merge1 = concatenate([output1, output2])
merge2 = Dense(5, activation='relu')(merge1)

last_output = Dense(1)(merge2)

model = Model(inputs=[input1, input2], outputs=last_output)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
import time
es = EarlyStopping(monitor='loss', patience=50,mode='min', verbose=1)

start_time = time. time()
model.fit([x1, x2], y, epochs=10000, batch_size=1, verbose=2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가, 예측
results = model.predict([x1_predict, x2_predict])
print('걸린시간 : ', end_time)
print(results)

'''
걸린시간 :  24.060146808624268
[[87.26017]]
'''