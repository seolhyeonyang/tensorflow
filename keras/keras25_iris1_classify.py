import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#! 다중 분류 문제

datasets = load_iris()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)     # (150, 4) (150,)
# print(y)


# 1. 데이터

#! 원핫인코딩 One-Hot-Encoding
# 수치로 단순 라벨링했을때 5가 1의 5배가 아닌 그냥 라벨링한것이다.

#^ 0 -> [1, 0, 0]
#^ 1 -> [0, 1, 0]
#^ 2 -> [0, 0, 1]        (150, ) -> (150, 3)

# [0, 1, 2, 1] ->
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 0],
#  [0, 1, 0]]     (4, ) -> (4, 3)

#! 라벨링 된 만큼 열의 개수로 된다.

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)       # 원핫인코딩 한것이다.

# print(y[:5])            # [[1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.], [1. 0. 0.]]
# print(y.shape)          # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=71)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape = (4,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))       #! 다중 분류 모델
#! softmax 를 활성화 함수로 사용한다. (출력값이 0~1 사이의 값으로 나오는데 총합이 1이 되야한다. 
#! -> 가장 큰값을 인정한다. [0.7, 0.2, 0.1] -> [1, 0, 0]로 인정한다.)
#^ output 3 개로 나와야 한다.(배열로 바꿨기 때문이다.)


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#! categorical_crossentropy - 다중분류 모델에서 사용

es = EarlyStopping(monitor = 'loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=3, callbacks=[es])

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

y_predict = model.predict(x_test[:5])
print(y_predict)
print(y_test[:5])


# [[8.4871276e-13 7.9120027e-06 9.9999213e-01]
#  [1.0000000e+00 3.6913788e-22 6.5814745e-14]
#  [2.1452975e-05 9.9970239e-01 2.7615411e-04]
#  [1.1473521e-04 3.9659228e-02 9.6022606e-01]
#  [7.7337263e-06 9.9987090e-01 1.2139664e-04]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]]



# loss :  0.0
# accuracy :  0.5

# 원핫인코딩 후
# loss :  0.07892175018787384
# accuracy :  0.9666666388511658
