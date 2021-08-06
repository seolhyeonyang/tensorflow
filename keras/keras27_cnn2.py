from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
#! Conv2D 는 4차원 데이터를 받아드린다.(= 이미지)


model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), 
                padding='same' ,input_shape=(10,10,1)))     # (N, 10, 10, 10)
model.add(Conv2D(20, (2,2), activation='relu'))             # (N, 9, 9, 20)
model.add(Conv2D(30, (2,2), padding='valid'))               # (N, 8, 8, 30)
model.add(MaxPool2D())                                   # (N, 4, 4, 30)
#! 데이터 shape 를 반으로 줄여준다. 데이터 손실이 발생한다.
#^ pooling 은 데이터를 겹쳐서 계산하지 않는다.(Conv2D처럼)
model.add(Conv2D(15, (2,2)))                                # (N, 3, 3, 15)
model.add(Flatten())                                        # (n, 135)
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 20)          820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 30)          2430
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 4, 4, 30)          0
_________________________________________________________________       # MaxPoolinge2D는 쪼갠 영역 안에서 특성(가장 큰 값)만 모아주는 것으로 데이터를 반으로 줄임
conv2d_3 (Conv2D)            (None, 3, 3, 15)          1815
_________________________________________________________________
flatten (Flatten)            (None, 135)               0
_________________________________________________________________
dense (Dense)                (None, 64)                8704
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 15,932
Trainable params: 15,932
Non-trainable params: 0
_________________________________________________________________
'''