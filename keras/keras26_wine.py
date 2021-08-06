import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# 1. 데이터
datasets = load_wine()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)         # (178, 13) (178,)
# print(x[:5])
# print(y)            # 0, 1, 2

y = to_categorical(y)

#print(y.shape)          # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

#scaler = MinMaxScaler()
#scaler = StandardScaler()
scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (13,)))
model.add(Dense(126, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=15, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=500, batch_size=5, callbacks=[es])


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test)


#TODO accuracy(=acc) 0.8이상 만들것
# MinMaxScaler
# loss :  0.012965155765414238
# accuracy :  1.0

#StandardScaler
# loss :  0.3907785415649414
# accuracy :  0.9444444179534912

# MaxAbsScaler
# loss :  4.632345735444687e-06
# accuracy :  1.0

#RobustScaler
# loss :  0.2270023226737976
# accuracy :  0.9722222089767456

# QuantileTransformer
# loss :  0.19392618536949158
# accuracy :  0.9722222089767456

# PowerTransformer
# loss :  0.35316991806030273
# accuracy :  0.9722222089767456