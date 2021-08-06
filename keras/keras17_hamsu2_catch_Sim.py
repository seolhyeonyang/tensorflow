'''
06_R2_2를 카피
함수형으로 리폼하시오.
서머리로 확인
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras import activations


x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred = [6]

# 완성한 뒤, 출력결과스샷

x = np.array(x)
y = np.array(y)

'''
model = Sequential()
model.add(Dense(1, input_dim = 1, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
'''
input1 = Input(shape=(1,))
dense1 = Dense(10)(input1)
dense2 = Dense(8)(dense1)
dense3 = Dense(6)(dense2)
dense4 = Dense(4)(dense3)
dense5 = Dense(2)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs = input1, outputs = output1)

# model.summary()


model.compile(loss='mse', optimizer='adam')


model.fit(x, y, epochs=100, batch_size=3, verbose=2)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print('r2스코어 : ', r2)

'''
r2스코어 :  0.8098569868556794
'''
