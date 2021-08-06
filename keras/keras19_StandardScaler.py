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

# 데이터 전처리

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# for문 안하고 미리 만들어 진것 사용

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)
# train 과 test의 scaler 이 다르다.

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)


''' 
scaler = MinMaxScaler()
scaler.fit(x_train)   # 훈련시키다. 실행 시키다.라는 의미
x_train = scaler.transform(x_train)   # 적용 시킨것
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
scaler.fit(x)   # 훈련시키다. 실행 시키다.라는 의미
x_scale = scaler.transform(x)   # 적용 시킨것 
전체 데이터를 transform 시키면 train에 과적합 되어 안되다. -> 산학회 질문

MinMaxScaler()는 데이터가 쏠려 있을때(한쪽으로 몰려있을때) 좋은 결과치를 얻기 힘든다.
'''



model = Sequential()
model.add(Dense(128, activation='relu', input_dim = 13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=99, batch_size=10)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print('y의 예측값 : ', y_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



'''
MinMaxSacler 전처리 후 fit(x_train)
loss :  7.9213738441467285
r2스코어 :  0.9196249534601572

StandardScaler 전처리 후 fit(x_train) train_size=0.7
loss :  6.691647052764893
r2스코어 :  0.9321024978045158

StandardScaler 전처리 후 fit(x_train) train_size=0.8
loss :  5.385702133178711
r2스코어 :  0.9328352630537192
'''