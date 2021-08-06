import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time


# 이전까지는 회귀 모델

datasets = load_breast_cancer()

# print(datasets.DESCR)       # 데이터 내용 (DESCR-묘사하다.)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (569, 30) (569,)

# print(y[:20])       # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y))     #[0 1]  y에 어떤 값이 있는지


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)      #(455, 30) (114, 30)

# 4차원 shape
# x_train = x_train.reshape(455, 10, 3, 1)
# x_test = x_test.reshape(114, 10, 3, 1)

# 3차원 shape
x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


# 2. 모델구성
model = Sequential()
# DNN 모델
# model.add(Dense(256, activation='relu', input_shape = (30,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#CNN 모델
# model.add(Conv2D(128,(2,1), padding='same', activation='relu', input_shape=(10, 3, 1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(8,(2,1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1, activation='sigmoid'))

#LSTM 모델
# model.add(LSTM(units=128, activation='relu', input_shape=(30,1)))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# Conv1D 모델
model.add(Conv1D(256, 2, activation='relu', input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


es = EarlyStopping(monitor = 'loss', patience=30, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#print('+'*10,' 예측 ', '+'*10)
y_predict = model.predict(x_test[:5])
# print(y_predict)
# print(y_test[:5])


""" 
++++++++++  예측  ++++++++++
[[1.0000000e+00]
 [2.5536369e-31]
 [1.0000000e+00]
 [1.0000000e+00]
 [7.7735827e-31]]
[1 0 1 1 0]



loss :  0.029148070141673088
r2 :  0.8734352379617311

이진분류모델 적용
loss :  0.447812557220459
accuracy :  0.9649122953414917

CNN
MinMaxScaler()
batch_size=10
shape(10, 3, 1)
걸린 시간 :  7.191290378570557
loss :  0.22203585505485535
accuracy :  0.9210526347160339

batch_size=1
걸린 시간 :  86.34942531585693
loss :  0.13131648302078247
accuracy :  0.9561403393745422

StandardScaler
batch_size=10
shape(10, 3, 1)
걸린 시간 :  14.295020580291748
loss :  0.14537513256072998
accuracy :  0.9561403393745422

batch_size=1
걸린 시간 :  70.61518335342407
loss :  0.21087291836738586
accuracy :  0.929824590682983

LSTM 모델
batch_size= 1
걸린 시간 :  1632.2622346878052
loss :  0.24231578409671783
accuracy :  0.9736841917037964

Conv1D 모델
걸린 시간 :  176.65825986862183
loss :  1.0619935989379883
accuracy :  0.9736841917037964
"""
