from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.datasets import cifar10, cifar100
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

# scaler = MinMaxScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 10) (10000, 10)

# 2. 모델
resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# resnet101.trainable = False

model = Sequential()
model.add(resnet101)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(516, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dense(10, activation='softmax'))
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_acc', patience=10, mode='auto', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=100, callbacks=[es], verbose=2, validation_split=0.2)
end_time = time.time() - start_time


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
#^ 1. cifar10
#? trainalbe = True 
#? FC
걸린 시간 :  2333.506377220154
loss :  1.259596347808838
accuracy :  0.7836999893188477

#? GAP
걸린 시간 :  1011.7290449142456
loss :  1.6565673351287842
accuracy :  0.7487000226974487

#? trainalbe = False 
#? FC
걸린 시간 :  476.22588777542114
loss :  1.804977536201477
accuracy :  0.38499999046325684

#? GAP
걸린 시간 :  480.1240785121918
loss :  1.816900372505188
accuracy :  0.392300009727478

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  989.9419419765472
loss :  3.6677961349487305
accuracy :  0.3837999999523163

#? GAP
걸린 시간 :  1002.0706670284271
loss :  2.937443256378174
accuracy :  0.4156999886035919

#? trainalbe = False 
#? FC
걸린 시간 :  454.7016775608063
loss :  3.9743151664733887
accuracy :  0.13860000669956207

#? GAP
걸린 시간 :  702.0255024433136
loss :  4.334958076477051
accuracy :  0.1370999962091446
'''