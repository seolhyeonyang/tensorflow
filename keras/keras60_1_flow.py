from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

#train_datagen = ImageDataGenerator(rescale = 1./255,)
#! 위 두개의 train_datagen 은 같은거다 밑에 파라미터들이 먹히지 않았기 때문이다.
#* flow에서 먹힌다.

#! 1. ImageDataGenerator를 정의
#! 2. 파일에서 땡겨올려면 -> flow_from_directory()
#! 3. 데이터에서 땡겨올려면 -> flow()

#^ flow_from_directory()
#^ x,y가 뭉쳐서 튜플 형식으로 반환된다. (입력 받는것도 튜플형태로 뭉쳐서 받는다.)

# xy_train = train_datagen.flow_from_directory(
#     '/study/_data/brain/train',
#     target_size=(150,150),
#     batch_size=5,
#     class_mode='binary',
#     shuffle=True
# )


#* flow()       (이미지가 수치화 되어있어야 한다.)
#* x,y 가 나눠져 있다.

#? 그림 1장을 100장 정도로 증폭
argument_size=50    
x_data = train_datagen.flow(
    np.tile((x_train[0]).reshape(28*28), argument_size).reshape(-1, 28, 28, 1),
    np.zeros(argument_size),
    batch_size=argument_size,
    shuffle=False
).next() 


#! iterator 방식으로 반환
#! .next() 해야 전체를 순환하면서 실행 (미사용시 하나만 실행)

print(type(x_data))
# <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
# next() -> <class 'tuple'>
print(type(x_data[0]))
# <class 'tuple'>
# next() -> <class 'numpy.ndarray'>
print(type(x_data[0][0]))
# <class 'numpy.ndarray'>
# next() -> <class 'numpy.ndarray'>
print(x_data[0][0].shape)
# (100, 28, 28, 1)
# next() -> (28, 28, 1)
print(x_data[0][1].shape)
# (100,)
print(x_data[0].shape)
# 오류 남 (튜플은 shape 할 수 없다.)
# next() -> (100, 28, 28, 1)
print(x_data[1].shape)
# next() -> (100,)


plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()