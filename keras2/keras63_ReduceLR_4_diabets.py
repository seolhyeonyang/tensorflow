# diabets 당뇨

import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(256, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))     


#3. 컴파일, 훈련
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

es = EarlyStopping(monitor= 'val_loss', patience=10, mode='min', verbose=1)

# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                     filepath='/study/_save/ModelCheckPoint/keras48_2_MCP_diabets.hdf5')

reduce_lr = ReduceLROnPlateau(monitor= 'val_loss', patience=5, mode='auto', verbose=1, factor=0.5)


start_time = time.time()

model.fit(x_train, y_train, epochs=600, batch_size=18, 
            validation_split=0.2, shuffle=True, verbose=2, callbacks=[es, reduce_lr])

# model.save('/study/_save/ModelCheckPoint/keras48_2_save_model_diabets.h5')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)


#model = load_model('/study/_save/ModelCheckPoint/keras48_2_save_model_diabets.h5')

# model = load_model('/study/_save/ModelCheckPoint/keras48_2_MCP_diabets.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse : ' , rmse)

# mse, R2

'''
걸린 시간 :  3.3227405548095703
loss :  2221.305908203125
r2스코어 :  0.5918198833116374
rmse :  47.130731892690385
'''