from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


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
# model.add(Dense(4, input_dim=3)) # inpu_dim은 2차원까지 쓴다. 2차원 넘어가면 쓸 수 없다.
model.add(Dense(4, input_shape=(3,))) 
#행 무시 열 우선 -> 특성(=열,피처,컬럼)만 넣는다. -> 행은 제일 앞에 온다. 나머지 input_shape가 된다.
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일 훈려
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=3000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred : ', result)

# 시각화
y_predict = model.predict(x)

plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.scatter(x[:,2], y)
plt.plot(x,y_predict, color='red')

plt.show()

'''

Epoch 3000

loss :  4.3772053004431655e-07
x_pred :  [[20.000807]]

'''