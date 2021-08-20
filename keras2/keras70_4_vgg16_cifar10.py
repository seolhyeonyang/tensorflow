from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
# model.add(Dense(124, activation='relu'))
# model.add(Dense(248, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# loss='sparse_categorical_crossentropy'

es = EarlyStopping(monitor= 'val_acc', patience=10, mode='auto', verbose=1)

model.fit(x_train, y_train, epochs=5000, batch_size=1000, callbacks=[es], verbose=2, validation_split=0.2)


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
#? batch_size=1000
#? vgg16.trainable = False
#? Flatten
#? MinMaxScaler
loss :  1.1679590940475464
accuracy :  0.6032999753952026

#? RobustScaler
loss :  1.107015609741211
accuracy :  0.6399999856948853

#? vgg16.trainable = True
#? Flatten
loss :  1.1351252794265747
accuracy :  0.802600026130676

#? vgg16.trainable = False
#? GAP
#? RobustScaler
loss :  1.0622049570083618
accuracy :  0.6284000277519226

#? vgg16.trainable = True
loss :  1.1241201162338257
accuracy :  0.7954000234603882
'''