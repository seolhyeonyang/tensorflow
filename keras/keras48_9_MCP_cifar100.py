# overfit를 극복하자
# 1. 전체 훈련 데이터가 많이 한다.
# 2. normailzation
# 3. dropout

import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


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

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

#print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()


# 2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Dropout, LSTM


""" model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D()) 

model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))               
model.add(MaxPool2D())

model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))                                                                
model.add(MaxPool2D())

model.add(GlobalAveragePooling2D())
# model.add(Flatten())  
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='/study/_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)


start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=1000, callbacks=[es, cp],validation_split=0.25, verbose=2)

model.save('/study/_save/ModelCheckPoint/keras48_9_save_model_cifar100.h5')

end_time = time.time() - start_time """

start_time = time.time()

#model = load_model('/study/_save/ModelCheckPoint/keras48_9_save_model_cifar100.h5')

model = load_model('/study/_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')

end_time = time.time() - start_time 

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



'''
걸린시간 :  2226.6264941692352
loss :  2.0916030406951904
accuracy :  0.49380001425743103

load_model
걸린시간 :  0.8742685317993164
loss :  2.0916030406951904
accuracy :  0.49380001425743103

check point
걸린시간 :  0.8276491165161133
loss :  1.9354218244552612
accuracy :  0.48980000615119934
'''