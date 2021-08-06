import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, Dropout, LSTM, concatenate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.convolutional import Conv1D
import tensorflow as tf
from datetime import datetime



# 1. 데이터

#! 데이터 불러오기
def file_open_csv(path, name):
    return pd.read_csv(f'{path}{name}주가 20210721.csv', encoding='EUC-KR') #encoding = 'cp949' 가능

f_path = '/study/samsung/_data/'
ss_name = '삼성전자 '
sk_name = 'SK'

ss = file_open_csv(f_path, ss_name)
sk =file_open_csv(f_path, sk_name)

#! 데이터 정리하기
def file_need_data(file):
    file = file.set_index('일자')
    new = file.loc[:'2011/01/03', ['고가', '저가','종가', '거래량', '시가']]
    new = new.sort_values(by =['일자'], axis=0)
    return new

ss_data = file_need_data(ss)
sk_data = file_need_data(sk)
#print(ss_data)

ss_x = ss_data.iloc[:, :4]
sk_x = sk_data.iloc[:, :4]

#print(ss_data.shape) #(2601, 5)
#print(sk_data)

#! 데이터 자르기(split)
ss_x_numpy = np.array(ss_x)
sk_x_numpy = np.array(sk_x)

ss_data_numpy = ss_data.values
sk_data_numpy = sk_data.values

def split_x(dataset, size):
    x = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size), :]
        x.append(subset)
    return np.array(x)

def split_y(dataset, size):
    y = []
    col = dataset[:,4]
    for i in range(len(dataset) - size + 1):
        n = i + size + 2
        if n < 2601:
            subset = col[n]
            y.append(subset)
        else:
            break
    return np.array(y)

days = 60

ss_x_data = split_x(ss_x_numpy, days)
sk_x_data = split_x(sk_x_numpy, days)

ss_y_data = split_y(ss_data_numpy, days)
#sk_y_data = split_y(sk_data_numpy, days)

#print(ss_x_data[:4,:])
#print(ss_y_data)

#print(ss_x_data, sk_x_data, ss_y_data)        #(2582, 20, 5) (2582, 20, 5) (2580,)

#print(ss_x_data.shape, sk_x_data.shape, ss_y_data.shape)        #(2582, 20, 5) (2582, 20, 5) (2580,)

#! x , y 데이터 train, test 나누기
x1 = ss_x_data[0:(ss_x_data.shape[0]-3), :]
x2 = sk_x_data[0:(sk_x_data.shape[0]-3), :]

y = ss_y_data
#y2 = sk_y_data

x1_pred = ss_x_data[(ss_x_data.shape[0]-1):, :]
x2_pred = sk_x_data[(sk_x_data.shape[0]-1):, :]

#print(x1,x2,y,x1_pred)
#print(x1.shape,x2.shape,y.shape,x1_pred.shape)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=78)

#print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)
# (2064, 20, 5) (516, 20, 5) (2064, 20, 5) (516, 20, 5) (2064,) (516,)

#! scaler 하기
x1_train1 = x1_train.reshape(x1_train.shape[0], (x1_train.shape[1] * x1_train.shape[2]))
x1_test1 = x1_test.reshape(x1_test.shape[0], (x1_test.shape[1] * x1_test.shape[2]))
x2_train1 = x2_train.reshape(x2_train.shape[0], (x2_train.shape[1] * x2_train.shape[2]))
x2_test1 = x2_test.reshape(x2_test.shape[0], (x2_test.shape[1] * x2_test.shape[2]))
x1_pred1 = x1_pred.reshape(x1_pred.shape[0], (x1_pred.shape[1] * x1_pred.shape[2]))
x2_pred1 = x2_pred.reshape(x2_pred.shape[0], (x2_pred.shape[1] * x2_pred.shape[2]))

#print(x1_train1.shape, x1_test1.shape, x2_train1.shape, x2_test1.shape, y_train.shape, y_test.shape)

scaler = MinMaxScaler()
#scaler = StandardScaler()
x1_train2 = scaler.fit_transform(x1_train1)
x1_test2 = scaler.transform(x1_test1)
x2_train2 = scaler.fit_transform(x2_train1)
x2_test2 = scaler.transform(x2_test1)
x1_pred2 = scaler.transform(x1_pred1)
x2_pred2 = scaler.transform(x2_pred1)

x1_train = x1_train2.reshape(x1_train.shape[0], x1_train.shape[1], x1_train.shape[2])
x1_test = x1_test2.reshape(x1_test.shape[0], x1_test.shape[1], x1_test.shape[2])
x2_train = x2_train2.reshape(x2_train.shape[0], x2_train.shape[1], x2_train.shape[2])
x2_test = x2_test2.reshape(x2_test.shape[0], x2_test.shape[1], x2_test.shape[2])
x1_pred = x1_pred2.reshape(x1_pred.shape[0], x1_pred.shape[1], x1_pred.shape[2])
x2_pred = x2_pred2.reshape(x2_pred.shape[0], x2_pred.shape[1], x2_pred.shape[2])

#print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)


# 2. 모델 구성

""" #^ samsung 모델
ss_input = Input(shape=(x1_train.shape[1], x1_train.shape[2]))
ss_lstm2 = LSTM(100, activation='relu')(ss_input)
ss_drop = Dropout(0.2)(ss_lstm2)
ss_dense4 = Dense(16, activation='relu')(ss_lstm2)
# ss_drop = Dropout(0.2)(ss_dense4)
# ss_dense5 = Dense(10, activation='relu')(ss_dense4)
# ss_drop = Dropout(0.2)(ss_dense5)
# ss_output = Dense(3, activation='relu')(ss_dense5)

#^ sk 모델
sk_input = Input(shape=(x2_train.shape[1], x2_train.shape[2]))
sk_lstm2 = LSTM(70, activation='relu')(sk_input)
sk_drop = Dropout(0.2)(sk_lstm2)
sk_dense2 = Dense(8, activation='relu')(sk_lstm2)
# sk_drop = Dropout(0.2)(sk_dense2)
# sk_output = Dense(5)(sk_dense2)

#^ 두 모델 앙상블
merge1 = concatenate([ss_dense4, sk_dense2])
merge_drop = Dropout(0.3)(merge1)
merge3 = Dense(7, activation='relu')(merge1)

#^ 최종 output
merge1_drop = Dropout(0.3)(merge3)
last_output = Dense(1)(merge3)

#^ 최종 모델
model = Model(inputs=[ss_input, sk_input], outputs=last_output)


# 컴파일, 훈련

#? 텐서보드
logdir="logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)
# tensorboard --logdir=./logs/fit/


#? 파일명_날짜_시간_val loss 저장
###################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')

filepath = '/study/samsung/_save2/'
filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
modelpath = ''.join([filepath, 'YSH06_', date_time, "-", filename])
###################################################

#? 컴파일
model.compile(loss = 'mae', optimizer='adam', metrics=['mse'])

#? earlystopping, mcp, 훈련하기
es = EarlyStopping(monitor='val_loss', patience=35, mode='auto', verbose=1, restore_best_weights=True)

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                    filepath= modelpath)

start_time = time.time()
model.fit([x1_train, x2_train], y_train, epochs=2000, batch_size=88, verbose=2, validation_split=0.15, callbacks=[es, cp, tensorboard_callback])
end_time = time.time() - start_time

#? 모델 저장하기
model.save('/study/samsung/_save2/YSH06.h5')

print('time : ', end_time)

# 4. 평가, 예측 

#* 기본 모델 실행
print('=================== 기본출력 ===================')

results = model.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model.predict([x1_pred, x2_pred])
print(y_predict) """

# #* load_model
print('=================== load_model ===================')
model2 = load_model('/study/samsung/_save2/YSH06.h5')

results = model2.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model2.predict([x1_pred, x2_pred])
print(y_predict)

# #* mcp
print('=================== check point ===================')
model3 = load_model('/study/samsung/_save2/YSH06_0723_1214-0033_4660.2563.hdf5')

results = model3.evaluate([x1_test, x2_test], y_test)
# print(results)
print("loss : ", results[0])

y_predict = model3.predict([x1_pred, x2_pred])
print(y_predict)


'''
loss :  4647.9453125
[[80003.68]]
'''