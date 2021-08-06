# np.save('/study/_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('/study/_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('/study/_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('/study/_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


# 1. 데이터
x_train = np.load('/study/_save/_npy/k59_3_train_x.npy')
y_train = np.load('/study/_save/_npy/k59_3_train_y.npy')
x_test = np.load('/study/_save/_npy/k59_3_test_x.npy')
y_test = np.load('/study/_save/_npy/k59_3_test_y.npy')


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

pred = model.predict(x_test)
print('pred : ', pred)