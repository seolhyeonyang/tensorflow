from tensorflow.keras.datasets import cifar10

# 이미지 32,32,3(칼라)


import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print('y[0] 값 : ', y_train[0])

plt.imshow(x_train[0], 'gray')

plt.show()