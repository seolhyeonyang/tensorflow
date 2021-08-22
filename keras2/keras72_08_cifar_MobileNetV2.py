from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
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
mobilenetV2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

mobilenetV2.trainable = False

model = Sequential()
model.add(mobilenetV2)
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
걸린 시간 :  235.1169991493225
loss :  1.6321446895599365
accuracy :  0.7408999800682068

#? GAP
걸린 시간 :  330.07207465171814
loss :  1.5547934770584106
accuracy :  0.7497000098228455

#? trainalbe = False 
#? FC
걸린 시간 :  69.19265699386597
loss :  2.287041187286377
accuracy :  0.2574999928474426

#? GAP
걸린 시간 :  67.7403473854065
loss :  2.2702176570892334
accuracy :  0.2614000141620636

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  692.8081867694855
loss :  4.895193576812744
accuracy :  0.29670000076293945

#? GAP
걸린 시간 :  729.1820385456085
loss :  4.316563129425049
accuracy :  0.3061000108718872

#? trainalbe = False 
#? FC
걸린 시간 :  109.38612151145935
loss :  5.0033674240112305
accuracy :  0.07919999957084656

#? GAP
걸린 시간 :  68.00246262550354
loss :  4.451751232147217
accuracy :  0.08349999785423279
'''