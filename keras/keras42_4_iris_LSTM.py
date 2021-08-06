import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time


#! 다중 분류 문제

datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150, 4) (150,)
# print(y)


# 1. 데이터

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)       # 원핫인코딩 한것이다.

# print(y[:5])            # [[1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.]]
# print(y.shape)          # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)

# 4차원 shape
# x_train = x_train.reshape(120, 2, 2, 1)
# x_test = x_test.reshape(30, 2, 2, 1)

# 3차원 shape
x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

# 2. 모델구성
model = Sequential()
# DNN 모델
# model.add(Dense(256, activation='relu', input_shape = (4,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# CNN 모델
# model.add(Conv2D(128, (1,1), activation='relu',padding='same', input_shape =(2,2,1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (1,1),padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (1,1),padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (1,1),padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(3, activation='softmax'))

# LSTM 모델
model.add(LSTM(units=64, activation='relu', input_shape=(4,1)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


es = EarlyStopping(monitor = 'loss', patience=30, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test[:5])
# print(y_predict)
# print(y_test[:5])


# [[8.4871276e-13 7.9120027e-06 9.9999213e-01]
#  [1.0000000e+00 3.6913788e-22 6.5814745e-14]
#  [2.1452975e-05 9.9970239e-01 2.7615411e-04]
#  [1.1473521e-04 3.9659228e-02 9.6022606e-01]
#  [7.7337263e-06 9.9987090e-01 1.2139664e-04]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]



# loss :  0.0
# accuracy :  0.5

# 원핫인코딩 후
# loss :  0.07892175018787384
# accuracy :  0.9666666388511658

'''
MinMaxScaler
shape(2, 2, 1)
batch_size=1
loss :  0.2677449882030487
accuracy :  0.8666666746139526

batch_size=8
걸린 시간 :  8.013680219650269
loss :  0.1775658279657364
accuracy :  0.9333333373069763

batch_size=10
걸린 시간 :  7.978341579437256
loss :  0.1604299545288086
accuracy :  0.8999999761581421

StandardScaler
걸린 시간 :  13.293134689331055
loss :  0.20298504829406738
accuracy :  0.9333333373069763

걸린 시간 :  6.165688991546631
loss :  0.16177912056446075
accuracy :  0.9333333373069763

걸린 시간 :  6.194804906845093
loss :  0.1607605367898941
accuracy :  0.9666666388511658

LSTM 모델
batch_size=1
걸린 시간 :  117.82172560691833
loss :  0.05033247172832489
accuracy :  0.9666666388511658
'''
