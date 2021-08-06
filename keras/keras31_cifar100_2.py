import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

#* 데이터 전처리

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

# ohe = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# ohe.fit(y_train)
# y_train = ohe.transform(y_train).toarray()
# y_test = ohe.transform(y_test).toarray()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=30, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(40, (3,3),padding='same', activation='relu'))             
model.add(Conv2D(50, (4,4), activation='relu'))               
model.add(MaxPool2D())
model.add(Conv2D(10, (2,2),padding='same', activation='relu'))
model.add(Conv2D(5, (2,2), activation='relu'))
model.add(MaxPool2D())                                                                
model.add(Flatten())                                        
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=5000, batch_size=10, callbacks=[es], verbose=2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
MinMaxScaler
loss :  12.736542701721191
accuracy :  0.26829999685287476

StandardScaler
loss :  13.576066970825195
accuracy :  0.25060001015663147

MaxAbsScaler
loss :  13.742348670959473
accuracy :  0.2547000050544739

RobustScaler
loss :  14.13158130645752
accuracy :  0.24729999899864197

QuantileTransformer
loss :  13.486981391906738
accuracy :  0.2594000101089477

PowerTransformer
loss :  13.367562294006348
accuracy :  0.2565999925136566
'''