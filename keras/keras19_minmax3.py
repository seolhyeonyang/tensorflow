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
print(np.min(x), np.max(x))         # 0.0 711.0
'''
 np는 수치, tensorflow 는 행렬 계산에 특화 되어 있다.
소수점 연산하는 것이 더 빠르다.
계산하기 쉽게 0~1사이 소수점으로 바꿔 계산한다. 
바꿔도 데이터 간의 비율은 변하지 않는다.
데이터 전처리(중에 하나) 라고 한다. (반든시 해야한다.)
데이터 폭발, 처리속도, 성능, 결과 등이 좋아진다.
/ max 하면 데이터를 0 과 1 사이의 값으로 만들 수 있다.
min max scalar(=정규화, =normalazion)라고 한다.
'''

# x = x/711       # max값을 따로 찾아 줘야 한다.
# x = x/np.max(x) # 기준점이 틀려져 비율이 달라져 값이 달라진다. (최소값이 0이 아닐때)

# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# (개체값 - 최소값)/ (최대값 - 최소값)하면 최소값은 0, 최대값은 1이 되면서 다른 데이터들은 0~1사이에 존재한다.
# 원본 데이터와 전처리 후 데이터가 같은 비율(데이터 간)을 유지 해야한다.
# min max scalar(= 0과1 사이로 정규화 시킨다.)
# 이 방식은 전체 데이터에서 최소값 , 최대값을 잡은것이다.
# 각 컬럼별 특성을 적용하지 않았다.(-> 각 컬럼의 minmax를 해줘야 한다. for 문을 이용해서 append 하면 된다.)

from sklearn.preprocessing import MinMaxScaler
# for문 안하고 미리 만들어 진것 사용

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=78)
# train 과 test의 scaler 이 다르다.


scaler = MinMaxScaler()
scaler.fit(x_train)   # 훈련시키다. 실행 시키다.라는 의미
x_train = scaler.transform(x_train)   # 적용 시킨것
x_test = scaler.transform(x_test)

''' 
scaler = MinMaxScaler()
scaler.fit(x)   # 훈련시키다. 실행 시키다.라는 의미
x_scale = scaler.transform(x)   # 적용 시킨것 
전체 데이터를 transform 시키면 train에 과적합 되어 안되다. -> 산학회 질문
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
minmax 전처리 후 fit(x)
loss :  17.176546096801758
r2스코어 :  0.8257163827778745

MinMaxSacler 전처리 후 fit(x)
loss :  7.832381248474121
r2스코어 :  0.920527922663305

MinMaxSacler 전처리 후 fit(x_train)
loss :  7.9213738441467285
r2스코어 :  0.9196249534601572
'''