from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([range(10)])

x = np.transpose(x)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = np.transpose(y) 

#완성하시오
x_pred = np.array([[9]])
print(x_pred.shape) # (1, 3)


# 2. 모델 구성
model = Sequential()
model.add(Dense(6, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))

# 3. 컴파일 훈려
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=5000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_pred)
print('x_pred : ', result)

# 시각화
y_predict = model.predict(x)

plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])
plt.plot(x, y_predict, color='red')
plt.show()

'''

Epoch 5000

loss :  0.005352909676730633
x_pred :  [[9.991501  1.5291562 0.9840712]]

'''