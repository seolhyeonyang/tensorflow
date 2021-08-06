from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.metrics import RootMeanSquaredError


x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred = [6]

# 완성한 뒤, 출력결과스샷

x = np.array(x)
y = np.array(y)

model = Sequential()
model.add(Dense(1, input_dim = 1, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

rmse = RootMeanSquaredError()

model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# mertics는 보조지표 ['보조지표1','보조지표2'], 중요한것은 loss이다.

model.fit(x, y, epochs=1000, batch_size=3)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('6의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print('r2스코어 : ', r2)

'''
과제 2
R2를 0.9 올려라
단톡에 스샷으로 인증 top3 커피
11일 (일) 밤 12시

r2스코어 :  0.8314773082598915

'''