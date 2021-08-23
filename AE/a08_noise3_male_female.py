# 실습, 과제
# kears61_5 남자 여자 데이터에 노이즈를 넣어
# 기미 주근깨 여드름 제거하시오!!

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import time
from matplotlib import pyplot as plt
import random


# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.25
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

x = np.load('/study2/_save/_npy/k59_5_men_women_x.npy')
y = np.load('/study2/_save/_npy/k59_5_men_women_y.npy')
x_pred = np.load('/study2/_save/_npy/k59_5_men_women_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

# print(x_train.shape, x_test.shape)          # (2647, 100, 100, 3) (662, 100, 100, 3)
# print(y_train.shape, y_test.shape)          # (2647,) (662,)

augment_size = 1353

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_argmented = x_train[randidx].copy()
y_argmented = y_train[randidx].copy()

# print(x_argmented.shape, y_argmented.shape)          # (1353, 100, 100, 3) (1353,)

x_argmented = train_datagen.flow(x_argmented, np.zeros(augment_size), batch_size=augment_size,
                                shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_argmented)) 
y_train = np.concatenate((y_train, y_argmented))

print(x_train.shape, y_train.shape)          # (4000, 100, 100, 3) (4000,)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. 모델
def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (2, 2), input_shape=(100, 100, 3),
                activation='relu', padding='same'))
    model.add(MaxPooling2D(1,1))
    model.add(Conv2D(hidden_layer_size/2, (2, 2), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(1,1)))
    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    return model

model = autoEncoder(hidden_layer_size=256)

# 3. 컴파일 / 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1) 

start_time = time.time()

model.fit(x_train_noised, x_train, epochs=100, verbose=2, validation_split=0.2, callbacks=[es])

end_time = time.time() - start_time                          

output = model.predict(x_test_noised)

# 시각화
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# original image
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(100, 100, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100, 100, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOiSED', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지르 아래 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(100, 100, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()