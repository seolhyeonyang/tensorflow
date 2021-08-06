import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#* 데이터 전처리

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


#print(np.unique(y_train))

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()



# 2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, LSTM, Conv1D

""" model = Sequential()
model.add(Conv2D(filters=20, kernel_size=(3,3), padding='same', input_shape=( 28, 28, 1)))
model.add(Conv2D(30, (3,3),padding='same', activation='relu'))             
model.add(Conv2D(40, (4,4), activation='relu'))               
model.add(MaxPool2D())
model.add(Conv2D(10, (2,2),padding='same', activation='relu'))
model.add(Conv2D(5, (2,2), activation='relu'))
model.add(MaxPool2D())                                                                
model.add(Flatten())                                        
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='/study/_save/ModelCheckPoint/keras48_7_MCP_fashion.hdf5')

es = EarlyStopping(monitor= 'loss', patience=20, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=150, callbacks=[es, cp],validation_split=0.2, verbose=2)

model.save('/study/_save/ModelCheckPoint/keras48_7_save_model_fashion.h5')

end_time = time.time() - start_time """

start_time = time.time()

#model = load_model('/study/_save/ModelCheckPoint/keras48_7_save_model_fashion.h5')

model = load_model('/study/_save/ModelCheckPoint/keras48_7_MCP_fashion.hdf5')

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
걸린 시간 :  385.8936734199524
loss :  1.0264495611190796
accuracy :  0.8974999785423279

load_model
걸린 시간 :  1.6342861652374268
loss :  1.0264495611190796
accuracy :  0.8974999785423279

check point
걸린 시간 :  0.9336395263671875
loss :  0.2895086109638214
accuracy :  0.9027000069618225
'''