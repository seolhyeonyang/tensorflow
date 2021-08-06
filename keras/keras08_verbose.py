from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.python.keras.utils.layer_utils import print_summary

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
print(x.shape) # (3, 10)

x = np.transpose(x) 
print(x.shape) # (10, 3)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10,)

#완성하시오
x_pred = np.array([[10, 1.3, 1]])
print(x_pred.shape) # (1, 3)


# 2. 모델 구성
model = Sequential()
model.add(Dense(4, input_dim=3))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일 훈려
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=3)
end = time.time() - start
print('걸린시간 : ', end)

'''
verbose
0 -> 과정 미출력
걸린시간 :  13.15369963645935

1 -> 과정 전부  출력 (디폴트)
걸린시간 :  19.727755308151245

2 -> 훈련횟수 / epoch  출력
걸린시간 :  15.050624132156372

3 -> epoch 만 출력
걸린시간 :  15.220284223556519

verbose = 1일때
batch=1, 10 인 경우 시간 측정
'''

#4. 평가, 예측

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred : ', result)


'''

Epoch 3000

loss :  4.3772053004431655e-07
x_pred :  [[20.000807]]

'''