# diabets 당뇨

import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(127, input_dim=10))
model.add(Dense(56))
model.add(Dense(36))
model.add(Dense(21))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, batch_size=20, 
            validation_split=0.3, shuffle=True, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y의 예측값 : ', y_test)


r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# mse, R2

'''
loss :  3196.669921875
r2스코어 :  0.5074499292314478
'''