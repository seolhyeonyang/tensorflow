'''
마지막 괄호는 생략

1) [1, 2, 3] -> 스칼라 3개 모여 벡터 1개가 되었다. -> (3, ) 라고 표현 행,열이 아니다.
2) [ [1, 2, 3] ] -> 1행 3열 -> (1,3)
3) [ [1, 2], [3, 4], [5, 6] ] -> (3,2)
4) [ [ [1, 2, 3], [4, 5, 6] ] ] -> (1,2,3)
5) [ [ [1,2], [3,4], [5,6] ] ] -> (1,3,2)
6) [ [[1],[2]], [[3],[4]] ] -> (2,2,1)
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) # (2,10) (행, 열)
print(x.shape)

x = np.transpose(x) 
print(x.shape) # (10, 2), 열을 맞춰야함(열=특성=피처=컬럼)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10,) 스칼라 10개로 이루어진 벡터 1개, dim=1

#완성하시오
x_pred = np.array([[10, 1.3]])
print(x_pred.shape) # (1, 2)

# 2. 모델 구성
model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일 훈려
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred : ', result)

# 시각화
y_predict = model.predict(x)

plt.scatter(x[:,0], y) # [:,n] n번째 열 추출, [n,:] n번째 행 추출
plt.scatter(x[:,1], y)
plt.plot(x,y_predict, color='red')

plt.show()

'''
Epoch 3300

loss :  4.855974289341702e-09
x_pred :  [[20.000065]]

'''