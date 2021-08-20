from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)

# scaler = MinMaxScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3) (50000, 10) (10000, 10)

# 2. 모델
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg16.trainable = False

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(516, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# loss='sparse_categorical_crossentropy'

es = EarlyStopping(monitor= 'val_acc', patience=10, mode='auto', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=1000, callbacks=[es], verbose=2, validation_split=0.2)
end_time = time.time() - start_time

print('걸린 시간 : ', end_time)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
#? batch_size=1000
#? vgg16.trainable = False
#? Flatten
#? MinMaxScaler
걸린 시간 :  75.58289670944214
loss :  2.620249032974243
accuracy :  0.367000013589859

#? RobustScaler
걸린 시간 :  80.20070886611938
loss :  2.6369516849517822
accuracy :  0.39570000767707825

#? vgg16.trainable = True
걸린 시간 :  395.2815544605255
loss :  5.441195487976074
accuracy :  0.38119998574256897

#? vgg16.trainable = False
#? GAP
걸린 시간 :  176.94754219055176
loss :  2.3517279624938965
accuracy :  0.4090000092983246

#? vgg16.trainable = True
걸린 시간 :  86.1335940361023
loss :  4.6053266525268555
accuracy :  0.009999999776482582
'''