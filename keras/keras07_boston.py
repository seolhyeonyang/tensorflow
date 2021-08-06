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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=78)


model = Sequential()
model.add(Dense(5, input_dim = 13))
model.add(Dense(13))
model.add(Dense(25))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=8000, batch_size=1)

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