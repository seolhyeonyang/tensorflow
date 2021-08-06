import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#! ImageDataGenerator -> 이미지 수치화 / 증폭
#! 지금은 수치화만 하고 있는것

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
#! test data는 검증해야한 data인데 증폭하면 data 손상을 줄 수 있다.
#! 해도 상관은 없는데 통상적으로 안한다.

xy_train = train_datagen.flow_from_directory(
    '/study/_data/brain/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# Found 160 images belonging to 2 classes.
#! x,y를 같이 만든다.
#! directory = 폴더
#! target_size 이미지의 크기를 동일한 크기로 만들어준다.
#! batch_size 하나로 묶일 이미지 개수
#! train밑에 2개의 폴더가 있는데, 하나는 정상, 하나는 비정상 폴더 자체가 라벨이 된다. class_mode='binary' 이진분류로 나눈다.

xy_test = test_datagen.flow_from_directory(
    '/study/_data/brain/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',
    shuffle=True
)
# Found 120 images belonging to 2 classes.
#! test는 train과 shape가 같아야 하므로 train -> test로만 바꿔준다.

#print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000019313849550>

#print(xy_train[0])

#print(xy_train[0][0])       # x값

#print(xy_train[0][1])       # y값
# [0. 1. 1. 0. 0.]
#! batch_size=5 으로 해서 5개로 나옴

#print(xy_train[0][0].shape, xy_train[0][1].shape)       # (5, 150, 150, 3) (5,)
#! batch_size, target_size, 컬러 (디폴트로 컬러로 받아들인다.)

print(xy_train[31][1])       # 마지막 배치 y
#print(xy_train[32][1])       # 없다.

#print(type(xy_train))
# <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
#print(type(xy_train[0]))
# <class 'tuple'>
#print(type(xy_train[0][0]))
# <class 'numpy.ndarray'>
#print(type(xy_train[0][1]))
# <class 'numpy.ndarray'>