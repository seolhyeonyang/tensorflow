import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)


print('#################### node 1개 시작 ####################')
model_01.compile(optimizer='adam', loss='binary_crossentropy')

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplot(7, 5, figsize=(15, 15))

# random_imgs = random.sample(range(output.shape[0]), 5)
# outputs = [x_test]

# for row_num, row in enumerate(axes):
#     for col_num, ax in enumerate(row):
#         ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap='gray')
#         ax.grid(False)
#         ax.set_xticks([])
#         ax.set_yticks([])
    
# plt.show()