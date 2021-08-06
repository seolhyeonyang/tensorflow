# overfit를 극복하자
# 1. 전체 훈련 데이터가 많이 한다.
# 2. normailzation
# 3. dropout

import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

#* 데이터 전처리

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#! train에 한해서 fit 과 transform을 한번에 가능하다.
x_train = scaler.fit_transform(x_train) 

x_test = scaler.transform(x_test)

# 4차원 shape
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)

# 3차원 shape
x_train = x_train.reshape(50000, 32 * 3, 32)
x_test = x_test.reshape(10000, 32 * 3, 32)

#print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Dropout, LSTM


model = Sequential()
# CNN 모델
# model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
# model.add(MaxPool2D()) 

# model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2),padding='same', activation='relu'))               
# model.add(MaxPool2D())

# model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2),padding='same', activation='relu'))                                                                
# model.add(MaxPool2D())

# model.add(GlobalAveragePooling2D())
# # model.add(Flatten())  
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))

# DNN 모델
# model.add(Dense(2048, input_shape =(32 * 32 * 3, ), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation='softmax'))

# LSTM 모델
model.add(LSTM(units=256, activation='relu', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=600, callbacks=[es],validation_split=0.25, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



'''
loss :  2.058232545852661
accuracy :  0.45649999380111694

DNN
Dropout
MinMaxScaler
걸린시간 :  224.36889243125916
loss :  5.0198798179626465
accuracy :  0.24719999730587006
'''