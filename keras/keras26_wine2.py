import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical



datasets = pd.read_csv('/study/_data/winequality-white.csv', sep= ';',
                        index_col=None, header=0)
#! ./ : 현재폴더
#! ../ : 상위폴더       
#* 파이참이랑 경로 구조가 다르다.

# print(datasets)
# print(datasets.shape)       # (4898, 12)
# print(datasets.info())
# print(datasets.describe())

#TODO 다중분류, 모델링하고 0.8이상으로 만들기

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

y = to_categorical(y)
#! to_categorical 은 0 부터 시작한다. 그래서 중간에 시작하는 것들에는 맞지 않다.


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=78)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# 2. 모델구성
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape = (11,)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(126, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=5000, batch_size=30, callbacks=[es])


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test)

'''
loss :  3.0570011138916016
accuracy :  0.668367326259613
'''