from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False
#! vgg16 훈련을 동결한다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

# model.trainable = False
#! model 훈련을 동결한다.

print(len(model.weights))               # 30
print(len(model.trainable_weights))     # 4

model.summary()
'''
=================================================================
Total params: 14,760,789
Trainable params: 46,101
Non-trainable params: 14,714,688
_________________________________________________________________
#! Trainable params: 46,101 만 훈련한다.
'''
