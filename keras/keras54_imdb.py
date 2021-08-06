from tensorflow.keras.datasets import imdb
import numpy as np


(x_train, y_trian), (x_test, y_test)= imdb.load_data(num_words=10000)

#print(y_trian)      # [1 0 0 ... 0 1 0]
#print(x_train.shape, x_test.shape)      # (25000,) (25000,)
#print(y_trian.shape, y_test.shape)      # (25000,) (25000,)

#print('뉴스기사의 최대길이 : ', max(len(i) for i in x_train))   # 2494
#print('뉴스기사의 평균길이 : ', sum(map(len, x_train))/ len(x_train))       # 238.71

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


x_train = pad_sequences(x_train, maxlen=240, padding='pre')
x_test = pad_sequences(x_test, maxlen=240, padding='pre')
#print(x_train.shape, x_test.shape)      # (25000, 240) (25000, 240)
#print(type(x_train), type(x_train[0]))      # <class 'numpy.ndarray'>
#print(x_train[0])

# y 확인
#print(np.unique(y_trian))       # [0 1]

# y_train = to_categorical(y_trian)
# y_test = to_categorical(y_test)
#print(y_trian.shape, y_test.shape)      # (25000,) (25000, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,  Dropout


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=120, input_length=240))
model.add(Dropout(0.5))
model.add(LSTM(100,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor= 'val_acc', patience=10, mode='auto', verbose=1)

import time

start_time = time.time()

model.fit(x_train, y_trian, epochs=500, batch_size=1500, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
batch_size=1000
걸린시간 :  142.7759087085724
loss :  0.6931546926498413
accuracy :  0.5

batch_size=1500
걸린시간 :  207.0262575149536
loss :  0.5357285737991333
accuracy :  0.746399998664856
'''