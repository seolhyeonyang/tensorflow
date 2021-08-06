from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


(x_train, y_train), (x_test,y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

#print(x_train[0], type(x_train[0]))     # type = list

print(y_train[0])   # 3
print(len(x_train[0]), len(x_train[1]))     # 87, 56
#! list는 shape가 안된다.
#! numpy는 데이터가 한개의 타입으로 통일 되어있어야, list는 여러 타입들이 들어 갈 수 있다.

print(x_train.shape, x_test.shape)      # (8982,) (2246,)
print(y_train.shape, y_test.shape)      # (8982,) (2246,)

print(type(x_train))        # <class 'numpy.ndarray'>

print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train))   # 2376
#print('뉴스기사의 최대길이 : ', max(len(x_train)))      # 실행 불가 안됨
print('뉴스기사의 평균길이 : ', sum(map(len, x_train))/ len(x_train))       #145.53

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape)      # (8982, 100) (2246, 100)
print(type(x_train), type(x_train[0]))      # <class 'numpy.ndarray'>
print(x_train[0])

# y 확인
print(np.unique(y_train))       # 0 ~ 45 으로 46개

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)      # (8982, 46) (2246, 46)


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=1000, input_length=100))
model.add(LSTM(800,  activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(46, activation='softmax'))


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor= 'val_acc', patience=10, mode='auto', verbose=1)

import time

start_time = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=2000, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# y_predict = model.predict(x_test)
# print(y_predict)