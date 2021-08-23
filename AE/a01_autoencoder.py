# 이미지작업
# encoder = 암호화
import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,  Input


input_img = Input(shape=(784, ))
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(1064, activation='relu')(input_img)
#! 특성이 강한것만 남고 약한건 사라진다. (중요한 특성만)
#^ 기미, 잡티 같은 약한 특성은 없어지고 눈, 코, 입만 남는다.

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
#! sigmoid는 0~1사이, relu는 양수 무한대... ,  값 범위가 넓어지면 흐려진다.

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
PS D:\study2> 
'''

autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)
#! 앞뒤가 똑같은 autoencoder (인풋 과 아웃풋이 같다.)
#^ y 라벨없이 x 값이 들어갔다가 x 가 나온다.

decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()