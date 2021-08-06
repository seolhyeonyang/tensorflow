# 실습
#TODO 훈련 데이터를 10만개로 증폭
#TODO 완료후 기존 모델과 비교
#TODO save_dir도 temp에 넣을 것


from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.10,
    shear_range=0.5,
    fill_mode='nearest',
)

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)

print(x_train.shape[0])     # 50000
print(randidx)              # [15361 48597 40898 ... 49863  2873  9910]
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)            # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

start_time = time.time()

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False
                                ).next()[0]

end_time = time.time() - start_time

# print(x_augmented[0][1].shape)
# print(x_augmented[0][1][:10])

print('걸린시간 : ', end_time)

#print(x_augmented.shape)        # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

x_train = x_train.reshape(100000, 784)
x_test = x_test.reshape(10000, 784)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(100000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# 2. 모델구성

model = Sequential()
model.add(Conv2D(filters=120, kernel_size=(3,3), padding='same', input_shape=( 28, 28, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(100, (3,3),padding='same', activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(80, (3,3),padding='same', activation='relu'))             
model.add(MaxPool2D())                                                              
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_acc', patience=20, mode='max', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=999, callbacks=[es], verbose=2, validation_split=0.2)

end_time = time.time() - start_time

print('time : ', end_time)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('============================= 평가 =============================')
print('loss : ', loss[0])
print('accuracy : ', loss[1])