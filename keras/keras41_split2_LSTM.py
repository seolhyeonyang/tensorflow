
#TODO
#     x               y
# 1, 2, 3, 4, 5       6
#, ...
# 95, 96, 97, 98, 99  100

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import time


# 1. 데이터
x_data = np.array(range(1,101))
x_predict = np.array(range(96, 106))

size_5 = 5
size_6 =6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size_6)

print(dataset)

x = dataset[:, :5]
y = dataset[:, 5]

# print("x : \n", x)
# print("y : ", y)

print(x.shape, y.shape)     # (95, 5)

x_predict_split = split_x(x_predict, size_6)
x_pred = x_predict_split[: , :-1]
# print(x_predict.shape)      (6, 4)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)      (76, 4) (20, 4)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



# 2. 모델구성
model = Sequential()
model.add(LSTM(units=10, activation='relu', input_shape=(4,1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='loss', patience=30, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=50, verbose=2, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('=' * 25)
print('걸린시간 : ', end_time)


y_pred = model.predict(x_test)

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse : ', rmse)

r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

'''
걸린시간 :  113.56092429161072
r2 :  0.9999772855910689
'''