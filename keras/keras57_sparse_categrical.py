import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#* 데이터 전처리

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

'''
ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()
'''
#! 원핫인코딩을 하면 loss = 'categorical_crossentropy'
#! 하지않으면 loss = 'sparse_categorical_crossentropy'

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=40, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), activation='relu'))             
model.add(Conv2D(5, (2,2), activation='relu'))               
model.add(MaxPool2D())
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(MaxPool2D())                                                                
model.add(Flatten())                                        
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=5000, batch_size=1000, callbacks=[es])


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

