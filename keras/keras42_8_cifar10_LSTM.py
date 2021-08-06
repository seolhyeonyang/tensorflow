import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
scaler.fit(x_train)
x_train = scaler.transform(x_train)
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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, LSTM

model = Sequential()
# CNN 모델
# model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(32, 32, 3)))
# model.add(Conv2D(30, (3,3),padding='same', activation='relu'))             
# model.add(Conv2D(40, (4,4), activation='relu'))               
# model.add(MaxPool2D())
# model.add(Conv2D(10, (2,2),padding='same', activation='relu'))
# model.add(Conv2D(5, (2,2), activation='relu'))
# model.add(MaxPool2D())                                                                
# model.add(Flatten())                                        
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# DNN 모델
# model.add(Dense(2048, input_shape =(32 * 32 * 3, ), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# LSTM 모델
model.add(LSTM(units=2058, activation='relu', input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=20, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=300, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
MinMaxScaler
loss :  4.057631015777588
accuracy :  0.6103000044822693

StandardScaler
loss :  3.780160903930664
accuracy :  0.6087999939918518

MaxAbsScaler
loss :  4.314354419708252
accuracy :  0.5997999906539917

RobustScaler
loss :  4.049912929534912
accuracy :  0.6410999894142151

QuantileTransformer
loss :  4.5399394035339355
accuracy :  0.6317999958992004

PowerTransformer

DNN 모델
RobustScaler
걸린 시간 :  170.30277681350708
loss :  4.230464458465576
accuracy :  0.5397999882698059

Dropout
걸린 시간 :  180.2485911846161
loss :  3.8453872203826904
accuracy :  0.5540000200271606

MinMaxScaler(Dropout)
걸린 시간 :  422.7895448207855
loss :  2.0860397815704346
accuracy :  0.5302000045776367

LSTM 모델

'''