'''
보스턴 지역 집값
'''

# 1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13)  input_dim = 13
print(y.shape) # (506,) output=1

print(datasets.feature_names) # 13열의 names 나옴 ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(datasets.DESCR) # 13열의 대한 상세 내용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 데이터 전처리

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(128, activation='relu', input_dim = 13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
#! 'loss'를 가지고 보는데, '최저' loss값이 '5번'까지 갱신되지 않을때 멈춘다.
# verbose = 1 멈춘 지점을 알려줌
# epoch = 19일때 stop하면 실제로는 5번 전인 14가 제일 좋은 값이다. 5번 밀려서 나온다.
# 최소값이 나올때마다 저장한다.

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

print(hist)
# <tensorflow.python.keras.callbacks.History object at 0x000001B9869E11C0>

print(hist.history.keys())
#dict_keys(['loss', 'val_loss'])

print('-'*10,' loss ','-'*10)
print(hist.history['loss'])

print('-'*10,' val_loss ','-'*10)
print(hist.history['val_loss'])

# 시각화
import matplotlib.pyplot as plt
#! x 가 시간 순서일때  x는 명시 안해도 된다. y값만 입력 가능하다.

from matplotlib import font_manager, rc

path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font)

plt.plot(hist.history['loss'])       # x : epoch / y : hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title('트레인 loss, val_loss')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss'])      # 범례생성, 순서대로 plot가 매치 (train loss -plt.plot(hist.history['loss']))

plt.show()

print('-'*10,' 평가예측 ','-'*10)
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
#! evaluate도 batch_size에 따라 나눠져서 평가된다.(디폴트 = 32)

y_predict = model.predict(x_test)
# print('y의 예측값 : ', y_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)




# MinMaxSacler 전처리 후 fit(x_train)
# loss :  7.9213738441467285
# r2스코어 :  0.9196249534601572

# StandardScaler 전처리 후 fit(x_train) train_size=0.7
# loss :  6.691647052764893
# r2스코어 :  0.9321024978045158

# StandardScaler 전처리 후 fit(x_train) train_size=0.8
# loss :  5.385702133178711
# r2스코어 :  0.9328352630537192

# Early Stopping
# loss
# epoch 58
# loss :  5.755970001220703
# r2스코어 :  0.9282176730850902

# verbose = 1
# Epoch 00047: early stopping
# 4/4 [==============================] - 0s 2ms/step - loss: 7.4745
# loss :  7.474547863006592
# r2스코어 :  0.906785406493656

# val_loss
# loss :  9.064138412475586
# r2스코어 :  0.8869617239948591
