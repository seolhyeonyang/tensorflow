import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train.shape, x_test.shape)      # (404, 13) (102, 13)
#print(y_train.shape, y_test.shape)      # (404,) (102,)

# 4차원 shape
# x_train = x_train.reshape(404, 13, 1, 1)
# x_test = x_test.reshape(102, 13, 1, 1)

# 3차원 shape
x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)



# 2. 모델 구성
model = Sequential()
# DNN 모델
# model.add(Dense(150, activation='relu', input_dim = 13))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1, activation='relu'))

# CNN 모델
# model.add(Conv2D(filters=32, kernel_size=(2,1), padding='same', activation='relu', input_shape=(13,1, 1)))
# model.add(Conv2D(64, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,1) , padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1))

# LSTM모델
model.add(LSTM(units=256, activation='relu', input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
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

print('=' * 25)
print('걸린시간 : ', end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)



'''
epochs = 93
batch_size = 10
train_size = 0.8

MinMaxScaler
loss :  7.141808032989502
r2 :  0.9109349800151908

StandardScaler
loss :  5.696831703186035
r2 :  0.9289551825391986

MaxAbsScaler
loss :  7.223166465759277
r2 :  0.9099203596554268

RobustScaler
loss :  5.604609489440918
r2 :  0.9301052847356391

QuantileTransformer
loss :  6.711169242858887
r2 :  0.9163054560948687

PowerTransformer
loss :  6.136422157287598
r2 :  0.9234730935556532

CNN 모델
1D
걸린시간 :  120.26649975776672
loss :  33.525733947753906
r2 :  0.5819027602598675

2D
걸린시간 :  182.8844187259674
loss :  39.143470764160156
r2 :  0.5118443515868927

걸린시간 :  140.06954073905945
loss :  21.34257698059082
r2 :  0.7338381053805523

걸린시간 :  156.28332138061523
loss :  17.191593170166016
r2 :  0.7856047927285774

LSTM 모델
batch_size=100
걸린시간 :  33.64995360374451
loss :  61.09859085083008
r2 :  0.23804349060164365

Dropout 0.2
걸린시간 :  19.790602684020996
loss :  63.680545806884766
r2 :  0.20584411021207638

batch_size=1
Dropout 0.2
걸린시간 :  1089.7856059074402
loss :  20.481924057006836
r2 :  0.7445712978802271
'''