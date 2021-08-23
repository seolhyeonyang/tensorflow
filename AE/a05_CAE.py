# 2번 복붙
# (CNN)딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적인 오토인코더
# 다른 하나는 딥하게 만든 구성
# 2개 성능 비교

'''
Conv2D
MaxPool
Conv2D
MaxPool
Conv2D -> encoder

Conv2D
UpSampling2D
Conv2D
UpSampling2D
Conv2D
UpSampling2D
Conv2D(1,)  -> Decoder
'''

import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float')/255

print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, UpSampling2D

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (2, 2), activation='relu', input_shape=(28, 28, 1)))
    model.add(UpSampling2D(2, 2))
    model.add(Conv2D(hidden_layer_size/2, (2, 2), activation='relu'))
    model.add(UpSampling2D(2, 2))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
