import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#! ImageDataGenerator -> 이미지 수치화 / 증폭
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '/study/_data/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    '/study/_data/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)


# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x_trian, y_trian)
hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32, validation_data=xy_test)
#* vaildation_steps = 4
#! xy가 쌍으로 있는 데이터는 fit_generator으로 한다.
#! steps_per_epoch 1번 epoch당 몇번 도는지 batch_size -> 160/5

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에 것 시각화할것

print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

