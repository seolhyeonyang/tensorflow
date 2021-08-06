# diabets 당뇨

import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
""" model = Sequential()
model.add(Dense(500, input_dim=10, activation='relu'))      #활성화 함수, 안써도 디폴트 갑이 있다. 지금은 relu가 성능이 좋다.
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))  """    # 현재는 마지막 레이어에 activation을 쓰지 않는다


#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam')

""" from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='/study/_save/ModelCheckPoint/keras47_MCP.hdf5')

start_time = time.time()

model.fit(x_train, y_train, epochs=600, batch_size=18, 
            validation_split=0.2, shuffle=True, verbose=2, callbacks=[es, cp])

model.save('/study/_save/ModelCheckPoint/keras47_MCP.h5')

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)"""

#model = load_model('/study/_save/ModelCheckPoint/keras47_model_save.h5')
#! save_model
model = load_model('/study/_save/ModelCheckPoint/keras47_MCP.hdf5')
#! check point

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
loss :  2072.823486328125
r2스코어 :  0.6191045086677045
rmse :  45.52827304512716

load_model
loss :  2072.823486328125
r2스코어 :  0.6191045086677045
rmse :  45.52827304512716

check point
loss :  2339.1123046875
r2스코어 :  0.5701721115163367
rmse :  48.364372385660076
'''