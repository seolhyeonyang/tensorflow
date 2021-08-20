from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
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
resNet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# resNet50.trainable = False

model = Sequential()
model.add(resNet50)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
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
걸린 시간 :  736.5938756465912
loss :  1.2594023942947388
accuracy :  0.765500009059906

#? GAP
걸린 시간 :  1044.1864290237427
loss :  1.2177903652191162
accuracy :  0.7782999873161316

#? trainalbe = False 
#? FC
걸린 시간 :  330.6549196243286
loss :  1.8044642210006714
accuracy :  0.47269999980926514

#? GAP
걸린 시간 :  197.74027276039124
loss :  1.5954334735870361
accuracy :  0.4745999872684479

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  666.3321504592896
loss :  3.5154874324798584
accuracy :  0.4677000045776367

#? GAP


#? trainalbe = False 
#? FC
걸린 시간 :  289.0565073490143
loss :  4.063207626342773
accuracy :  0.17810000479221344

#? GAP
걸린 시간 :  345.54128551483154
loss :  4.324460029602051
accuracy :  0.1868000030517578
'''