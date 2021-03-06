import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#* 데이터 전처리

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train))

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()



# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
# CNN 모델
# model.add(Conv2D(filters=20, kernel_size=(3,3), padding='same', input_shape=( 28, 28, 1)))
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
model.add(Dense(1024, input_shape =(28 *28, ), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=20, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=150, callbacks=[es],validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
DNN 모델
걸린 시간 :  103.96652817726135
loss :  0.8209357261657715
accuracy :  0.8968999981880188

걸린 시간 :  166.12305903434753
loss :  0.886321485042572
accuracy :  0.8939999938011169

Dropout
걸린 시간 :  225.04321599006653
loss :  0.8146324753761292
accuracy :  0.8967999815940857
'''