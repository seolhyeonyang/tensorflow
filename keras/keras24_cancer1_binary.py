import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#! 이진 분류 모델
# 이전까지는 회귀 모델

datasets = load_breast_cancer()

# print(datasets.DESCR)       # 데이터 내용 (DESCR-묘사하다.)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (569, 30) (569,)

print(y[:20])       # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
print(np.unique(y))     #[0 1]  y에 어떤 값이 있는지


# 1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))       #! 이진 분류 모델
#! sigmoid 는 출력값을 0~1사이 값으로 출력해 준다. (이진분류에서 활성화 함수로 이용한다.)


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#! binary_crossentropy - 이진분류 모델에서 사용

es = EarlyStopping(monitor = 'loss', patience=5, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=10, callbacks=[es])

# 시각화

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])

# plt.title('loss, val_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss, val_loss'])

# plt.show()


# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print('+'*10,' 예측 ', '+'*10)
y_predict = model.predict(x_test[:5])
print(y_predict)
print(y_test[:5])


# ++++++++++  예측  ++++++++++
# [[1.0000000e+00]
#  [2.5536369e-31]
#  [1.0000000e+00]
#  [1.0000000e+00]
#  [7.7735827e-31]]
# [1 0 1 1 0]



# loss :  0.029148070141673088
# r2 :  0.8734352379617311

# 이진분류모델 적용
# loss :  0.447812557220459
# accuracy :  0.9649122953414917
