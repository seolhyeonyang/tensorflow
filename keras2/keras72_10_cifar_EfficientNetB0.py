from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.datasets import cifar10, cifar100
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
efficientNetB0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

efficientNetB0.trainable = False

model = Sequential()
model.add(efficientNetB0)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(516, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.add(Dense(100, activation='softmax'))


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
걸린 시간 :  303.71972608566284
loss :  4.483490467071533
accuracy :  0.12960000336170197

#? GAP
걸린 시간 :  499.6788594722748
loss :  4.716355323791504
accuracy :  0.23849999904632568

#? trainalbe = False 
#? FC
걸린 시간 :  189.52320337295532
loss :  2.302611827850342
accuracy :  0.10000000149011612

#? GAP
걸린 시간 :  101.84154868125916
loss :  2.302600383758545
accuracy :  0.10000000149011612

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  396.24738240242004
loss :  9.96066951751709
accuracy :  0.02459999918937683

#? GAP
걸린 시간 :  780.2446773052216
loss :  29.714981079101562
accuracy :  0.015200000256299973

#? trainalbe = False 
#? FC
걸린 시간 :  101.59241604804993
loss :  4.605479717254639
accuracy :  0.009999999776482582

#? GAP
걸린 시간 :  102.15156745910645
loss :  4.605474948883057
accuracy :  0.009999999776482582
'''