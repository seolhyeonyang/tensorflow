from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


x = [1,2,3,4,5]
y = [1,2,4,3,5]
x_pred = [6]

# 완성한 뒤, 출력결과스샷

x = np.array(x)
y = np.array(y)

model = Sequential()
model.add(Dense(1, input_dim = 1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=8500, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('6의 예측값 : ', result)