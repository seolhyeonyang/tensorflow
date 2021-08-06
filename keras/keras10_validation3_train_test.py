from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=True, random_state=66)
x_test, y_test, x_val, y_val = train_test_split(x_test, y_test, train_size=0.5, shuffle=True, random_state=66)
# val 까지 하려면 2번 나눠주면 된다. 먼저 train 과 test를 나누고 test를 다시 test와 val로 나눠준다.

'''
x_train = np.array([1,2,3,4,5,6,7])     # 훈련, 공부하는거
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])             # 평가하는거
y_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_val = np.array([11,12,13])            # 검증하는거(머신 스스로)  val_loss가 더 loss가 더 좋다.
y_val = np.array([11,12,13])
'''
'''
#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))

# 통상적으로 loos가 val_loss가 더 안좋게 나온다. 
# 하지만 val_loss를 염두해 두고 해야 한다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([11])
print('x의 예측값 : ', y_predict)
'''

'''
Epoch 10

loss :  3.666698932647705
11의 예측값 :  [[10.560807]]
'''