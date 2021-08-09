import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
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

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)



# 2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (4,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                     filepath='/study/_save/ModelCheckPoint/keras48_4_MCP_iris.hdf5')

es = EarlyStopping(monitor = 'val_accuracy', patience=30, mode='auto', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor= 'val_accuracy', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es, reduce_lr], validation_split=0.2, verbose=2)


end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test[:5])
# print(y_predict)
# print(y_test[:5])



'''
걸린 시간 :  10.32819652557373
loss :  0.10616479068994522
accuracy :  0.9333333373069763

걸린 시간 :  8.422664165496826
loss :  0.041311342269182205
accuracy :  1.0
'''
