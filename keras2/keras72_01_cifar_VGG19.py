# 실습
# cifar10 과 cifar100 으로 모델 만들것
# trainalbe = True. False
# FC로 만든것과 Avarage Pooling 으로 만든것 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
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
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg19.trainable = False

model = Sequential()
model.add(vgg19)
# model.add(Flatten())
# model.add(Dense(516, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(GlobalAveragePooling2D())
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
걸린 시간 :  759.0401520729065
loss :  1.2260968685150146
accuracy :  0.803399980068206

#? GAP
걸린 시간 :  578.907488822937
loss :  1.1051031351089478
accuracy :  0.7975000143051147

#? trainalbe = False 
#? FC
걸린 시간 :  94.69227123260498
loss :  1.8590595722198486
accuracy :  0.6341999769210815

#? GAP
걸린 시간 :  180.26313853263855
loss :  1.0744235515594482
accuracy :  0.6261000037193298

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  914.5772407054901
loss :  4.880942344665527
accuracy :  0.34380000829696655

#? GAP
걸린 시간 :  987.7533831596375
loss :  4.1617279052734375
accuracy :  0.3472999930381775

#? trainalbe = False 
#? FC
걸린 시간 :  93.88689470291138
loss :  3.4109132289886475
accuracy :  0.3626999855041504

#? GAP
걸린 시간 :  182.121648311615
loss :  2.471031427383423
accuracy :  0.3847000002861023
'''