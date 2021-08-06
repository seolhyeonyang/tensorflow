import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
x_train = np.load('/study/_save/_npy/k59_8_cat_dog_x_train.npy')
y_train = np.load('/study/_save/_npy/k59_8_cat_dog_y_train.npy')
x_test = np.load('/study/_save/_npy/k59_8_cat_dog_x_test.npy')
y_test = np.load('/study/_save/_npy/k59_8_cat_dog_y_test.npy')


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', input_shape=(200, 200, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1, restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

print('시간은 :', end_time)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# 4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('============================= 평가 =============================')
print('loss : ', loss[0])
print('acc : ', acc[1])

temp = model.predict(x_test)

temp = tf.argmax(temp, axis=1)

temp = pd.DataFrame(temp)

temp.to_csv('/study/_save/_csv/cat_dog.csv')