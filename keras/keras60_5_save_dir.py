from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
import time


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


augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)

# print(x_train.shape[0])     # 60000
# print(randidx)              # [32783 55783 44318 ... 49994 20121 46465]
# print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
#print(x_augmented.shape)            # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

start_time = time.time()

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False,
                                save_to_dir='/study/temp/') # .next()[0]

end_time = time.time() - start_time

print(x_augmented[0][1].shape)
print(x_augmented[0][1][:10])
#! .next() 안하면 x_augmented를 사용 할때 마다 flow를 실행한다.
#! .next()[0]은 x_augmented[0]이라는 이야기다.

print('걸린시간 : ', end_time)
exit()
#print(x_augmented.shape)        # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
#! 리스트 합치는 것 append
#! 넘파이는 concatenate

#print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)

#print(x_augmented[0][27])

plt.figure(figsize=(2,2))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')

    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')

plt.show()