import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#! ImageDataGenerator -> 이미지 수치화 / 증폭

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
    batch_size=200,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    '/study/_data/brain/test',
    target_size=(150,150),
    batch_size=200,
    class_mode='binary',
    shuffle=True
)
# Found 120 images belonging to 2 classes.
#! batch_size를 크게 줘서 데이터를 한곳에 몰아 넣는다. (나누지 않고)

print(xy_train[0][0].shape, xy_train[0][1].shape)       # (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)         # (120, 150, 150, 3) (120,)

np.save('/study/_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('/study/_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('/study/_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
np.save('/study/_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])