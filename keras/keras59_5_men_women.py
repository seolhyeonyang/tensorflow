# 실습 1.
#TODO men women 데이터로 모델링을 구성할 것

# 실습 2.
#TODO 본인 사진으로 predict 하시오!     d:\data 안에 본인 사진 넣고 확인

import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터 구성
# train_datagen = ImageDataGenerator(
#     rescale = 1./255,
#     horizontal_flip= True,
#     vertical_flip= True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest'
# )

""" datagen = ImageDataGenerator(rescale=1./255)

xy_data = datagen.flow_from_directory(
    '/study/_data/men_women',
    target_size=(100,100),
    batch_size=4000,
    class_mode='binary',
    shuffle=True
)

# Found 3309 images belonging to 2 classes.

pred = datagen.flow_from_directory(
    '/study/_data/men_women_pred',
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    shuffle=True
)

#print(xy_data[0][0].shape, xy_data[0][1].shape)       # (3309, 100, 100, 3) (3309,)

#print(pred[0][0].shape, pred[0][1].shape)
np.save('/study/_save/_npy/k59_5_men_women_x.npy', arr=xy_data[0][0])
np.save('/study/_save/_npy/k59_5_men_women_y.npy', arr=xy_data[0][1])
np.save('/study/_save/_npy/k59_5_men_women_pred.npy', arr=pred[0][0]) """


x = np.load('/study/_save/_npy/k59_5_men_women_x.npy')
y = np.load('/study/_save/_npy/k59_5_men_women_y.npy')
pred = np.load('/study/_save/_npy/k59_5_men_women_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

#print(x_train.shape, x_test.shape)      # (2647, 100, 100, 3) (662, 100, 100, 3)

# print(pred.shape)

# print(np.unique(y_train))
# print(np.unique(y_test))

x_train = x_train.reshape(2647, 30000)
x_test = x_test.reshape(662, 30000)
pred = pred.reshape(1, 30000)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(2647, 100, 100, 3)
x_test = x_test.reshape(662, 100, 100, 3)
pred = pred.reshape(1, 100, 100, 3)


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(128, (2,2), padding='same', input_shape=(100, 100, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (2,2), activation='relu'))             
model.add(MaxPooling2D(2,2))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=20, mode='auto', verbose=1, restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, steps_per_epoch=99, validation_split=0.2,validation_steps=20, verbose=2, callbacks=[es])
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

pred_2 = model.predict(pred)

temp = tf.argmax(pred_2, axis=1)
print(temp)