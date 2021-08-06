from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
#! Conv2D 는 4차원 데이터를 받아드린다.(= 이미지)


model = Sequential()                                            # (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5,5,1)))   # (N, 4, 4, 10)
#! 가로세로 5 x 5 1개(input_shape=(5,5,1)) 를 
#! 가지고 2 x 2 (kernel_size=(2,2)) 로 잘라서  
#! 4 x 4로 되는데 아웃풋이 10이라서

model.add(Conv2D(20, (2,2),activation='relu'))                  # (N, 3, 3, 20)
#! kernerl_size= 생략가능 (아웃풋 뒤에 바로 작성)
#^ input_shape=(batch_size, height, weight, color)
#^ 행 (= 전체 데이터 크기 = batch_size )무시

model.add(Flatten())                                            # (N, 180)
#! 데이터를 펴주는 것

model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 3, 3, 20)          820
_________________________________________________________________   # Flatten 은 그냥 데이터를 펴주는 것이라 Param 이 0 이다.
flatten (Flatten)            (None, 180)               0
_________________________________________________________________   # (180 +1) x 64 = 11,584
dense (Dense)                (None, 64)                11584
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 14,567
Trainable params: 14,567
Non-trainable params: 0
_________________________________________________________________
'''


model = Sequential()                                            
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(5,5,1)))
#! padding = 'same' 하면 데이터 shape 가 줄어들지 않고 그대로 간다.
# padding 디폴트는 valid 이다.

model.add(Conv2D(20, (2,2),activation='relu'))                                    
model.add(Flatten())                                            
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()