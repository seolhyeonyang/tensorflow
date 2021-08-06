from tokenize import endpats
import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Conv1D, Flatten, Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터 구성
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)      #(353, 10) (89, 10)

# 4차원 shape
# x_train = x_train.reshape(353, 10, 1, 1)
# x_test = x_test.reshape(89, 10, 1, 1)

# 3차원 shape
x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)


# 2. 모델구성
model = Sequential()
# DNN 모델
# model.add(Dense(128, activation='relu', input_dim = 10))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation='relu'))

# CNN 모델
# model.add(Conv2D(128,(2,1), padding='same', activation='relu', input_shape=(10, 1, 1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(8,(2,1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1))

# LSTM모델
# model.add(LSTM(units=512, activation='relu', input_shape=(10,1)))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# Conv1D 모델
model.add(Conv1D(128, 2, activation='relu', input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor= 'loss', patience=20, mode='min', verbose=1)

start_time = time.time()

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=2, validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

'''
epochs = 99
batch_size = 13
train_size = 0.8

MinMaxScaler
loss :  2368.189697265625
r2 :  0.5648289877291146

StandardScaler
loss :  2380.068359375
r2 :  0.5626461721981587

CNN
MinMaxScaler
shape(353, 5, 2, 1)
batch_size = 13
걸린 시간 :  16.04354500770569
loss :  4286.60400390625
r2 :  0.2123072722921956

batch_size = 1
걸린 시간 :  241.87857031822205
loss :  2528.65087890625
r2 :  0.5353431096489447

shape(353, 10, 1, 1)
걸린 시간 :  46.84613609313965
loss :  3896.49658203125
r2 :  0.28399211221225673

StandardScaler
shape(353, 5, 2, 1)
걸린 시간 :  214.93317341804504
loss :  3234.443359375
r2 :  0.405648900807967

shape(353, 10, 1, 1)
걸린 시간 :  122.69089818000793
loss :  2601.67138671875
r2 :  0.5219251157140123

LSTM모델
batch_size=100
걸린 시간 :  20.79532265663147
loss :  4169.86279296875
r2 :  0.2337592344044609

batch_size=1
Dropout 0.2
걸린 시간 :  1183.26700258255
loss :  3436.31103515625
r2 :  0.36855431953602436

Conv1D 모델
걸린 시간 :  22.54615545272827
loss :  2161.8408203125
r2 :  0.6027470265619028
'''