import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time


datasets = pd.read_csv('/study/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)

# print(datasets)
# print(datasets.shape)       # (4898, 12)
# print(datasets.info())
# print(datasets.describe())


# 1. 데이터
#^ 1. 판다스 -> 넘파이

datasets = datasets.to_numpy()
#datasets = datasets.values #도 가능


#^ 2. x와 y를 분리

# x = datasets.iloc[ : , :11]     # (4898, 11), df 데이터 나누기
# y = datasets.iloc[:, 11:]       # (4898, 1)

x = datasets[ : , :11]      
y = datasets[:, 11:]

# print(x.shape)       # (4898, 11)
# print(y.shape)      # (4898, 1)

#^ 3. y의 라벨을 확인 np.unique(y)

print(np.unique(y))     # [3. 4. 5. 6. 7. 8. 9.]

# 데이터 나누기

#y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

#print(x_train.shape, x_test.shape)      # (3918, 11) (980, 11)

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 4차원 shape
# x_train = x_train.reshape(3918, 11, 1, 1)
# x_test = x_test.reshape(980, 11, 1, 1)

# 3차원 shape
x_train = x_train.reshape(3918, 11, 1)
x_test = x_test.reshape(980, 11, 1)

# print(y_train.shape, y_test.shape)        #(3918, 7) (980, 7)

# 2. 모델구성
model = Sequential()
# DNN 모델
# model.add(Dense(2048, activation='relu', input_shape = (11,)))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(126, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# CNN 모델
# model.add(Conv2D(1024,(2,1), padding='same', activation='relu', input_shape=(11, 1, 1)))
# model.add(Dropout(0.2))
# model.add(Conv2D(512,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(128,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64,(2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32,(2,1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(7, activation='softmax'))

# LSTM 모델
# model.add(LSTM(units=1024, activation='relu', input_shape=(11,1)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(7, activation='softmax'))


# Conv1D 모델
model.add(Conv1D(1024, 2, activation='relu', input_shape=(11, 1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=30, mode='min', verbose=1)

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=30, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

#y_predict = model.predict(x_test)

'''
loss :  3.0570011138916016
accuracy :  0.668367326259613

MinMaxScaler
shape(11, 1, 1)
batch_size=15
걸린시간 :  527.3549416065216
loss :  1.2833858728408813
accuracy :  0.6071428656578064

batch_size=30
걸린시간 :  762.3887486457825
loss :  1.6425259113311768
accuracy :  0.6234694123268127

batch_size=100
걸린시간 :  85.20309162139893
loss :  1.0991426706314087
accuracy :  0.5520408153533936

StandardScaler
batch_size=30
걸린시간 :  215.7723264694214
loss :  1.45803701877594
accuracy :  0.6051020622253418

LSTM 모델
batch_size=30
걸린시간 :  765.1585824489594
loss :  3.350297689437866
accuracy :  0.6102041006088257

Conv1D 모델
걸린시간 :  123.40996646881104
loss :  2.1854732036590576
accuracy :  0.640816330909729
'''