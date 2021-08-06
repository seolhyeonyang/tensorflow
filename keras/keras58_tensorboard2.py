from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

tb = TensorBoard(log_dir='/study/_save/_graph', histogram_freq=0, write_graph=True,
                write_images=True)

model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2, callbacks=[tb], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print('6의 예측값 : ', result)



'''
Epoch 8000

loss :  0.38000601530075073
6의 예측값 :  [[5.700907]]
'''