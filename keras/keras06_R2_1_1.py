from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

'''
print(x_test)
print(y_test)


np.random.shuffle(x)
np.random.shuffle(y)

x_train = x[0:7] # 0부터 7번째 전까지
y_train = y[:7]
x_test = x[-3:]
y_test = y[7:]

x_train = np.array(x[:7])
y_train = np.array(y[:7])
x_test = np.array(x[7:10])
y_test = np.array(y[7:10])
같은 의미

x_train = x[0:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[70:]

print(x_train.shape, y_train.shape) # (70,) (70,)
print(x_test.shape, y_test.shape) # (30,) (30,)
'''

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('x의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('r2스코어 : ', r2)