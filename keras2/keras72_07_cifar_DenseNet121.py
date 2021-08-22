from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
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
densenet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

densenet121.trainable = False

model = Sequential()
model.add(densenet121)
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
걸린 시간 :  1037.359675168991
loss :  0.859603226184845
accuracy :  0.8234999775886536

#? GAP
걸린 시간 :  916.4517557621002
loss :  0.8788977265357971
accuracy :  0.815500020980835

#? trainalbe = False 
#? FC
걸린 시간 :  180.52158284187317
loss :  1.815149188041687
accuracy :  0.6582000255584717

#? GAP
걸린 시간 :  168.75437378883362
loss :  1.8326125144958496
accuracy :  0.659500002861023

#^ 2. cifar100
#? trainalbe = True 
#? FC
걸린 시간 :  839.5628578662872
loss :  3.2167258262634277
accuracy :  0.43540000915527344

#? GAP
걸린 시간 :  1059.0956594944
loss :  2.757173538208008
accuracy :  0.5382000207901001

#? trainalbe = False 
#? FC
걸린 시간 :  208.64234018325806
loss :  4.650302410125732
accuracy :  0.36880001425743103

#? GAP
걸린 시간 :  182.55675435066223
loss :  4.188840389251709
accuracy :  0.3756999969482422
'''