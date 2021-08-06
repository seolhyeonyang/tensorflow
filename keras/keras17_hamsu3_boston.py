'''
보스텀을 함수형으로 구현하시오
서머리 확인
'''

'''
보스턴 지역 집값
'''

from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13)  input_dim = 13
print(y.shape) # (506,) output=1

print(datasets.feature_names) # 13열의 names 나옴 ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(datasets.DESCR) # 13열의 대한 상세 내용

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=31)

input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)
dense2 = Dense(13)(dense1)
dense3 = Dense(25)(dense2)
dense4 = Dense(16)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(5)(dense5)
output1 = Dense(1)(dense6)

model = Model(inputs = input1, outputs = output1)

model.summary()

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=10)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y의 예측값 : ', y_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
loss :  24.130126953125
r2스코어 :  0.7551611166723715
'''