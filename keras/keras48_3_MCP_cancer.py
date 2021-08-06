import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time


# 이전까지는 회귀 모델

datasets = load_breast_cancer()

# print(datasets.DESCR)       # 데이터 내용 (DESCR-묘사하다.)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

#print(x.shape, y.shape)     # (569, 30) (569,)

# print(y[:20])       # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
# print(np.unique(y))     #[0 1]  y에 어떤 값이 있는지


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#print(x_train.shape, x_test.shape)      #(455, 30) (114, 30)


""" # 2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath='/study/_save/ModelCheckPoint/keras48_3_MCP_cancer.hdf5')

es = EarlyStopping(monitor = 'loss', patience=30, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=1000, batch_size=15, callbacks=[es, cp], validation_split=0.2, verbose=2)

model.save('/study/_save/ModelCheckPoint/keras48_3_save_model_cancer.h5')

end_time = time.time() - start_time """

start_time = time.time()

#model = load_model('/study/_save/ModelCheckPoint/keras48_3_save_model_cancer.h5')

model = load_model('/study/_save/ModelCheckPoint/keras48_3_MCP_cancer.hdf5')

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
걸린 시간 :  109.75471472740173
loss :  0.4663536250591278
accuracy :  0.9736841917037964

load_model
걸린 시간 :  0.8491077423095703
loss :  0.4663536250591278
accuracy :  0.9736841917037964

check point
걸린 시간 :  0.8143501281738281
loss :  0.09906867146492004
accuracy :  0.9736841917037964
"""
