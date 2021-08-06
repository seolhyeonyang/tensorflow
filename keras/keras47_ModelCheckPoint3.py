import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate  
from sklearn.model_selection import train_test_split


x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001,1101))
# y = np.array([range(1001,1101)])
# y = np.transpose(y)

# print(x1.shape, x2.shape, y.shape)     # (100, 3) (100, 3) (100,)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, shuffle=True, random_state=66)


# 2-1. 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3) 
# x1, y1 모델

# 2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)
# x2, y1 모델


merge1 = Concatenate()([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1)(merge3) 

model = Model(inputs = [input1, input2], outputs=last_output)

# model.summary()



#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)

###################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')

filepath = '/study/_save/ModelCheckPoint/'
filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
modelpath = ''.join([filepath, 'k47_', date_time, "-", filename])
###################################################

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath= modelpath)

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es, cp])

model.save('/study/_save/ModelCheckPoint/keras49_EarlyStopping.h5')

from sklearn.metrics import r2_score
print('=================== 기본출력 ===================')

#4 평가, 예측 
results = model.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print('=================== load_model ===================')
model2 = load_model('/study/_save/ModelCheckPoint/keras49_EarlyStopping.h5')

results = model2.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model2.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

print('=================== check point ===================')
model3 = load_model('/study/_save/ModelCheckPoint/keras49_EarlyStopping.hdf5')

results = model3.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model3.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



'''
EarlyStoppint( restore_best_weights=False )  -> 밀린지점에서 저장
=================== 기본출력 ===================
1/1 [==============================] - 0s 16ms/step - loss: 1876.9420 - mse: 1876.9420
loss :  1876.9420166015625
r2스코어 :  -1.1465921654939915
=================== load_model ===================
1/1 [==============================] - 0s 99ms/step - loss: 1876.9420 - mse: 1876.9420
loss :  1876.9420166015625
r2스코어 :  -1.1465921654939915
=================== check point ===================
1/1 [==============================] - 0s 96ms/step - loss: 1987.2104 - mse: 1987.2104
loss :  1987.21044921875
r2스코어 :  -1.2727022495155476


EarlyStoppint( restore_best_weights=True )  -> 멈춘지점에서 저장
=================== 기본출력 ===================
1/1 [==============================] - 0s 15ms/step - loss: 2173.7610 - mse: 2173.7610
loss :  2173.760986328125
r2스코어 :  -1.486053407272943
=================== load_model ===================
1/1 [==============================] - 0s 96ms/step - loss: 2173.7610 - mse: 2173.7610
loss :  2173.760986328125
r2스코어 :  -1.486053407272943
=================== check point ===================
1/1 [==============================] - 0s 96ms/step - loss: 2173.7610 - mse: 2173.7610
loss :  2173.760986328125
r2스코어 :  -1.486053407272943
'''